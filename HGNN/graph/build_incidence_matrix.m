function H = build_incidence_matrix(hyperedges, num_nodes)
% BUILD_INCIDENCE_MATRIX  Incidence matrix H Generation => Eventually we
% will amke this to dynamic hyper-edge generation. Currently let's just
% keep it as the static hyper-edge.
%
% Input:
%   hyperedges : cell array, each cell means the node indices that belongs to one hyper-edge
%                예) {[1,2,3], [2,4,5], [1,3,5]}
%   num_nodes  : the number of entire nodes(N)
%
% Return:
%   H : (N x E) sparse incidence matrix
%       When H(i,j) = 1, node i belongs to hyper-edge j.

% -----------------------------------------------------------------------
% 0. Basic information Extraction
% -----------------------------------------------------------------------
num_edges = length(hyperedges);   % The number of hyper-edges E

% -----------------------------------------------------------------------
% 1. Index collections to compose the Sparse Matrix.
% -----------------------------------------------------------------------
row_idx = [];   % node index
col_idx = [];   % hyper-edge index

for e = 1:num_edges
    nodes = hyperedges{e}(:)';      % nodes that belong to this hyper-edge
    if isempty(nodes)
        continue;
    end
    if any(nodes < 1) || any(nodes > num_nodes) || any(nodes ~= floor(nodes))
        error('build_incidence_matrix: 하이퍼엣지 %d에 잘못된 노드 인덱스가 있습니다.', e);
    end
    row_idx = [row_idx, nodes];     %#ok<AGROW>
    col_idx = [col_idx, repmat(e, 1, length(nodes))];  %#ok<AGROW>
end

% -----------------------------------------------------------------------
% 2. Sparse incidence matrix Generation
% -----------------------------------------------------------------------
H = sparse(row_idx, col_idx, ones(size(row_idx)), num_nodes, num_edges);

% -----------------------------------------------------------------------
% 3. Validity Test (TODO: Extend it if we need it.)
% -----------------------------------------------------------------------
% - 노드 인덱스 범위 확인
% - 빈 하이퍼엣지 확인
% - 고립 노드(degree=0) 경고

end
