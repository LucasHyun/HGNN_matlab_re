function [X, H, Y_true, train_mask, val_mask, test_mask] = load_data(dataset_name, options)
% LOAD_DATA  Load and preprocess a dataset
%
% Inputs:
%   dataset_name : string ('toy' | 'cora' | 'custom')
%   options      : dataset-specific option struct; optional
%
% Outputs:
%   X          : (N x F)   node-feature matrix
%   H          : (N x E)   incidence matrix (sparse)
%   Y_true     : (N x C)   one-hot labels
%   train_mask : (N x 1)   training-node mask
%   val_mask   : (N x 1)   validation-node mask
%   test_mask  : (N x 1)   test-node mask

if nargin < 2 || isempty(options)
    options = struct();
end

switch lower(dataset_name)

    % -------------------------------------------------------------------
    case 'toy'   % Small synthetic dataset for implementation checks
    % -------------------------------------------------------------------
        rng(7);

        num_nodes  = 12;
        F          = 5;    % Feature dimension
        C          = 3;    % Number of classes

        % Class-centered features make HGNN behavior easier to verify than random labels.
        labels = [ones(4, 1); 2 * ones(4, 1); 3 * ones(4, 1)];
        centers = [2 0 0 1 0;
                   0 2 0 0 1;
                   0 0 2 1 1];

        X = zeros(num_nodes, F);
        for c = 1:C
            class_idx = find(labels == c);
            X(class_idx, :) = repmat(centers(c, :), length(class_idx), 1) ...
                              + 0.05 * randn(length(class_idx), F);
        end

        % Hyperedge definitions
        hyperedges = {[1,2,3,4], [5,6,7,8], [9,10,11,12], ...
                      [1,2,3], [6,7,8], [10,11,12]};
        H = build_incidence_matrix(hyperedges, num_nodes);

        % One-hot labels
        Y_true = full(sparse(1:num_nodes, labels', 1, num_nodes, C));

        % Fixed split that includes train/validation/test nodes for each class.
        train_mask = false(num_nodes, 1);  val_mask = false(num_nodes, 1);  test_mask = false(num_nodes, 1);
        train_mask([1,2,5,6,9,10]) = true;
        val_mask([3,7,11])         = true;
        test_mask([4,8,12])        = true;

    % -------------------------------------------------------------------
    case 'cora'  % Load the real Cora citation dataset
    % -------------------------------------------------------------------
        data_dir = fullfile(fileparts(mfilename('fullpath')), 'cora');
        content_file = fullfile(data_dir, 'cora.content');
        cites_file = fullfile(data_dir, 'cora.cites');

        if ~exist(content_file, 'file') || ~exist(cites_file, 'file')
            error(['load_data: Cora files are missing. The following files are required:\n' ...
                   '  %s\n  %s'], content_file, cites_file);
        end

        [paper_ids, X, labels] = read_cora_content(content_file);
        H = build_cora_hypergraph(cites_file, paper_ids, options);
        Y_true = labels_to_onehot(labels);
        [train_mask, val_mask, test_mask] = split_masks(labels);

    % -------------------------------------------------------------------
    case 'custom'  % User-defined data
    % -------------------------------------------------------------------
        % TODO: replace this branch with the desired custom data format.
        error('load_data: implement the custom loader before using dataset_name="custom".');

    otherwise
        error('load_data: unknown dataset "%s"', dataset_name);
end

end

% -----------------------------------------------------------------------
% Cora helper functions
% -----------------------------------------------------------------------
function [paper_ids, X, labels] = read_cora_content(content_file)
    fid = fopen(content_file, 'r');
    if fid < 0
        error('read_cora_content: cannot open file: %s', content_file);
    end
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);

    lines = lines{1};
    num_nodes = numel(lines);
    first_parts = strsplit(strtrim(lines{1}));
    num_features = numel(first_parts) - 2;

    paper_ids = cell(num_nodes, 1);
    raw_labels = cell(num_nodes, 1);
    row_idx = [];
    col_idx = [];

    for i = 1:num_nodes
        parts = strsplit(strtrim(lines{i}));
        if numel(parts) ~= num_features + 2
            error('read_cora_content: line %d has an unexpected number of columns.', i);
        end

        paper_ids{i} = parts{1};
        raw_labels{i} = parts{end};

        feature_values = str2double(parts(2:end-1));
        nz = find(feature_values ~= 0);
        row_idx = [row_idx, repmat(i, 1, numel(nz))]; %#ok<AGROW>
        col_idx = [col_idx, nz]; %#ok<AGROW>
    end

    X = sparse(row_idx, col_idx, 1, num_nodes, num_features);
    X = normalize_rows(X);

    class_names = unique(raw_labels);
    label_map = containers.Map(class_names, num2cell(1:numel(class_names)));
    labels = zeros(num_nodes, 1);
    for i = 1:num_nodes
        labels(i) = label_map(raw_labels{i});
    end
end

function H = build_cora_hypergraph(cites_file, paper_ids, options)
% BUILD_CORA_HYPERGRAPH  Build research-group hyperedges around cited papers
%
%   Frequently cited papers are treated as research-group seeds.
%   Each hyperedge is {seed paper} union {papers that cite the seed paper}.
%
%   options.min_group_citations : minimum citation count for a seed (default 5)
%   options.max_research_groups : max number of top citation-count seeds (default inf)
%   options.include_singletons  : add nodes outside all research groups as singleton
%                                 hyperedges (default true)

    num_nodes = numel(paper_ids);
    id_map = containers.Map(paper_ids, num2cell(1:num_nodes));
    if nargin < 3 || isempty(options)
        options = struct();
    end

    min_group_citations = get_option(options, 'min_group_citations', 5);
    min_group_citations = max(1, floor(min_group_citations));
    max_research_groups = get_option(options, 'max_research_groups', inf);
    include_singletons  = get_option(options, 'include_singletons', true);

    fid = fopen(cites_file, 'r');
    if fid < 0
        error('build_cora_hypergraph: cannot open file: %s', cites_file);
    end
    cite_pairs = textscan(fid, '%s%s');
    fclose(fid);

    cited_ids  = cite_pairs{1};
    citing_ids = cite_pairs{2};

    % -------------------------------------------------------------------
    % 1. Collect citation context: cited paper -> list of citing papers
    % -------------------------------------------------------------------
    cite_map = containers.Map();   % key: cited node index as a string
                                   % val: array of citing node indices
    citation_count = zeros(num_nodes, 1);

    for i = 1:numel(cited_ids)
        if ~isKey(id_map, cited_ids{i}) || ~isKey(id_map, citing_ids{i})
            continue;
        end
        cited  = id_map(cited_ids{i});
        citing = id_map(citing_ids{i});
        citation_count(cited) = citation_count(cited) + 1;

        key = num2str(cited);
        if isKey(cite_map, key)
            cite_map(key) = [cite_map(key), citing];
        else
            cite_map(key) = citing;
        end
    end

    % -------------------------------------------------------------------
    % 2. Select research-group seeds among frequently cited papers
    % -------------------------------------------------------------------
    seed_nodes = find(citation_count >= min_group_citations);
    [~, order] = sort(citation_count(seed_nodes), 'descend');
    seed_nodes = seed_nodes(order);

    if isfinite(max_research_groups)
        max_research_groups = max(0, floor(max_research_groups));
        seed_nodes = seed_nodes(1:min(max_research_groups, numel(seed_nodes)));
    end

    % -------------------------------------------------------------------
    % 3. Build research-group hyperedges: {seed paper} union {citing papers}
    % -------------------------------------------------------------------
    hyperedges = cell(numel(seed_nodes), 1);
    for i = 1:numel(seed_nodes)
        cited_node = seed_nodes(i);
        key = num2str(cited_node);
        if isKey(cite_map, key)
            citers = cite_map(key);
        else
            citers = [];
        end
        hyperedges{i} = unique([cited_node, citers]);
    end

    % -------------------------------------------------------------------
    % 4. Handle isolated nodes by adding singleton hyperedges
    % -------------------------------------------------------------------
    if include_singletons
        covered = false(num_nodes, 1);
        for i = 1:numel(hyperedges)
            covered(hyperedges{i}) = true;
        end
        isolated = find(~covered);
        for i = 1:numel(isolated)
            hyperedges{end+1} = isolated(i); %#ok<AGROW>
        end
    end

    H = build_incidence_matrix(hyperedges, num_nodes);
end

function Y = labels_to_onehot(labels)
    num_nodes = numel(labels);
    num_classes = max(labels);
    Y = full(sparse(1:num_nodes, labels(:)', 1, num_nodes, num_classes));
end

function [train_mask, val_mask, test_mask] = split_masks(labels)
    rng(7);

    num_nodes = numel(labels);
    num_classes = max(labels);
    train_per_class = 20;
    val_count = 500;
    test_count = 1000;

    train_mask = false(num_nodes, 1);
    val_mask = false(num_nodes, 1);
    test_mask = false(num_nodes, 1);

    for c = 1:num_classes
        class_idx = find(labels == c);
        class_idx = class_idx(randperm(numel(class_idx)));
        take = min(train_per_class, numel(class_idx));
        train_mask(class_idx(1:take)) = true;
    end

    remaining = find(~train_mask);
    remaining = remaining(randperm(numel(remaining)));

    val_count = min(val_count, numel(remaining));
    val_idx = remaining(1:val_count);
    val_mask(val_idx) = true;

    remaining = remaining(val_count + 1:end);
    test_count = min(test_count, numel(remaining));
    test_idx = remaining(1:test_count);
    test_mask(test_idx) = true;
end

function X_norm = normalize_rows(X)
    row_sum = sum(X, 2);
    row_sum(row_sum == 0) = 1;
    X_norm = spdiags(1 ./ row_sum, 0, size(X, 1), size(X, 1)) * X;
end

function val = get_option(options, field, default)
    if isstruct(options) && isfield(options, field) && ~isempty(options.(field))
        val = options.(field);
    else
        val = default;
    end
end
