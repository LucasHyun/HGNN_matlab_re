function [Theta_conv, D_v, D_e, W] = compute_laplacian(H, edge_weights)
% COMPUTE_LAPLACIAN  HGNN 정규화 전파 행렬 계산
%
% 핵심 수식:
%   Theta_conv = D_v^(-1/2) * H * W * D_e^(-1) * H' * D_v^(-1/2)
%
% 입력:
%   H            : (N x E) incidence matrix (sparse 권장)
%   edge_weights : (E x 1) 하이퍼엣지 가중치 벡터 (생략 시 모두 1)
%
% 출력:
%   Theta_conv : (N x N) 정규화된 전파 행렬
%   D_v        : (N x N) 노드 degree 대각행렬
%   D_e        : (E x E) 하이퍼엣지 degree 대각행렬
%   W          : (E x E) 하이퍼엣지 가중치 대각행렬

[num_nodes, num_edges] = size(H);

% -----------------------------------------------------------------------
% 1. 하이퍼엣지 가중치 행렬 W
% -----------------------------------------------------------------------
if nargin < 2 || isempty(edge_weights)
    edge_weights = ones(num_edges, 1);   % 기본값: 균등 가중치
end
edge_weights = edge_weights(:);
if numel(edge_weights) ~= num_edges
    error('compute_laplacian: edge_weights 길이는 하이퍼엣지 수와 같아야 합니다.');
end
W = spdiags(edge_weights, 0, num_edges, num_edges);     % (E x E)

% -----------------------------------------------------------------------
% 2. 노드 degree 행렬 D_v
%    d_v(i) = sum_e W(e,e) * H(i,e)  (가중치 반영)
% -----------------------------------------------------------------------
d_v = H * edge_weights;                  % (N x 1)
D_v = spdiags(d_v, 0, num_nodes, num_nodes);            % (N x N)

% -----------------------------------------------------------------------
% 3. 하이퍼엣지 degree 행렬 D_e
%    d_e(e) = 해당 하이퍼엣지에 속한 노드 수
% -----------------------------------------------------------------------
d_e = sum(H, 1)';                        % (E x 1)
D_e = spdiags(d_e, 0, num_edges, num_edges);            % (E x E)

% -----------------------------------------------------------------------
% 4. D_v^(-1/2), D_e^(-1) 계산 (0 degree 노드 처리 포함)
% -----------------------------------------------------------------------
dv_invsqrt = d_v .^ (-0.5);
dv_invsqrt(isinf(dv_invsqrt)) = 0;      % degree=0인 노드 처리
Dv_invsqrt = spdiags(dv_invsqrt, 0, num_nodes, num_nodes);  % (N x N)

de_inv = d_e .^ (-1);
de_inv(isinf(de_inv)) = 0;              % 빈 하이퍼엣지 처리
De_inv = spdiags(de_inv, 0, num_edges, num_edges);      % (E x E)

% -----------------------------------------------------------------------
% 5. 정규화 전파 행렬 조합
%    Theta_conv = Dv^(-1/2) * H * W * De^(-1) * H' * Dv^(-1/2)
% -----------------------------------------------------------------------
Theta_conv = Dv_invsqrt * H * W * De_inv * H' * Dv_invsqrt;

end
