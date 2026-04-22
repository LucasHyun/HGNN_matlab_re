function H = build_incidence_matrix(hyperedges, num_nodes)
% BUILD_INCIDENCE_MATRIX  Build the sparse incidence matrix H.
%
% Input:
%   hyperedges : cell array; each cell contains the node indices for one hyperedge.
%                Example: {[1,2,3], [2,4,5], [1,3,5]}
%   num_nodes  : total number of nodes (N)
%
% Return:
%   H : (N x E) sparse incidence matrix
%       When H(i,j) = 1, node i belongs to hyper-edge j.

% -----------------------------------------------------------------------
% 0. Basic information extraction
% -----------------------------------------------------------------------
num_edges = length(hyperedges);   % The number of hyper-edges E

% -----------------------------------------------------------------------
% 1. Collect indices for sparse matrix construction.
% -----------------------------------------------------------------------
row_idx = [];   % node index
col_idx = [];   % hyper-edge index

for e = 1:num_edges
    nodes = hyperedges{e}(:)';      % nodes that belong to this hyper-edge
    if isempty(nodes)
        continue;
    end
    if any(nodes < 1) || any(nodes > num_nodes) || any(nodes ~= floor(nodes))
        error('build_incidence_matrix: hyperedge %d contains an invalid node index.', e);
    end
    row_idx = [row_idx, nodes];     %#ok<AGROW>
    col_idx = [col_idx, repmat(e, 1, length(nodes))];  %#ok<AGROW>
end

% -----------------------------------------------------------------------
% 2. Sparse incidence matrix generation
% -----------------------------------------------------------------------
H = sparse(row_idx, col_idx, ones(size(row_idx)), num_nodes, num_edges);

% -----------------------------------------------------------------------
% 3. Validity checks (TODO: extend if needed)
% -----------------------------------------------------------------------
% - Check node-index ranges.
% - Check empty hyperedges.
% - Warn about isolated nodes with degree zero.

end
