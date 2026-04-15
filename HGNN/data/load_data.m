function [X, H, Y_true, train_mask, val_mask, test_mask] = load_data(dataset_name)
% LOAD_DATA  데이터셋 로드 및 전처리
%
% 입력:
%   dataset_name : 문자열 ('toy' | 'cora' | 'custom')
%
% 출력:
%   X          : (N x F)   노드 피처 행렬
%   H          : (N x E)   incidence matrix (sparse)
%   Y_true     : (N x C)   one-hot 레이블
%   train_mask : (N x 1)   학습 노드 마스크
%   val_mask   : (N x 1)   검증 노드 마스크
%   test_mask  : (N x 1)   테스트 노드 마스크

switch lower(dataset_name)

    % -------------------------------------------------------------------
    case 'toy'   % 간단한 합성 데이터 (구현 테스트용)
    % -------------------------------------------------------------------
        rng(7);

        num_nodes  = 12;
        F          = 5;    % 피처 차원
        C          = 3;    % 클래스 수

        % 클래스별 중심을 둔 피처. 랜덤 레이블보다 HGNN 동작 확인에 적합합니다.
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

        % 하이퍼엣지 정의
        hyperedges = {[1,2,3,4], [5,6,7,8], [9,10,11,12], ...
                      [1,2,3], [6,7,8], [10,11,12]};
        H = build_incidence_matrix(hyperedges, num_nodes);

        % 레이블 (one-hot)
        Y_true = full(sparse(1:num_nodes, labels', 1, num_nodes, C));

        % 클래스마다 train/val/test가 포함되도록 고정 분할
        train_mask = false(num_nodes, 1);  val_mask = false(num_nodes, 1);  test_mask = false(num_nodes, 1);
        train_mask([1,2,5,6,9,10]) = true;
        val_mask([3,7,11])         = true;
        test_mask([4,8,12])        = true;

    % -------------------------------------------------------------------
    case 'cora'  % 실제 Cora citation dataset 로드
    % -------------------------------------------------------------------
        data_dir = fullfile(fileparts(mfilename('fullpath')), 'cora');
        content_file = fullfile(data_dir, 'cora.content');
        cites_file = fullfile(data_dir, 'cora.cites');

        if ~exist(content_file, 'file') || ~exist(cites_file, 'file')
            error(['load_data: Cora 파일이 없습니다. 다음 파일이 필요합니다:\n' ...
                   '  %s\n  %s'], content_file, cites_file);
        end

        [paper_ids, X, labels] = read_cora_content(content_file);
        H = build_cora_hypergraph(cites_file, paper_ids);
        Y_true = labels_to_onehot(labels);
        [train_mask, val_mask, test_mask] = split_masks(labels);

    % -------------------------------------------------------------------
    case 'custom'  % 사용자 정의 데이터
    % -------------------------------------------------------------------
        % TODO: 원하는 형식으로 교체
        error('load_data: custom 로더를 직접 구현하세요.');

    otherwise
        error('load_data: 알 수 없는 데이터셋 "%s"', dataset_name);
end

end

% -----------------------------------------------------------------------
% Cora helper functions
% -----------------------------------------------------------------------
function [paper_ids, X, labels] = read_cora_content(content_file)
    fid = fopen(content_file, 'r');
    if fid < 0
        error('read_cora_content: 파일을 열 수 없습니다: %s', content_file);
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
            error('read_cora_content: %d번째 줄의 column 수가 올바르지 않습니다.', i);
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

function H = build_cora_hypergraph(cites_file, paper_ids)
    num_nodes = numel(paper_ids);
    id_map = containers.Map(paper_ids, num2cell(1:num_nodes));

    fid = fopen(cites_file, 'r');
    if fid < 0
        error('build_cora_hypergraph: 파일을 열 수 없습니다: %s', cites_file);
    end
    cite_pairs = textscan(fid, '%s%s');
    fclose(fid);

    cited_ids = cite_pairs{1};
    citing_ids = cite_pairs{2};
    row_idx = [];
    col_idx = [];

    for i = 1:numel(cited_ids)
        if isKey(id_map, cited_ids{i}) && isKey(id_map, citing_ids{i})
            cited = id_map(cited_ids{i});
            citing = id_map(citing_ids{i});

            row_idx = [row_idx, cited, citing]; %#ok<AGROW>
            col_idx = [col_idx, citing, cited]; %#ok<AGROW>
        end
    end

    A = sparse(row_idx, col_idx, 1, num_nodes, num_nodes);
    H = spones(A + speye(num_nodes));
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
