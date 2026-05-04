function [Theta_conv, D_v, D_e, W] = compute_propagation_matrix(H, edge_weights)
% COMPUTE_PROPAGATION_MATRIX  Compute the HGNN normalized propagation matrix
%
% Core equation:
%   Theta_conv = D_v^(-1/2) * H * W * D_e^(-1) * H' * D_v^(-1/2)
%
% Inputs:
%   H            : (N x E) incidence matrix; sparse format is recommended.
%   edge_weights : (E x 1) hyperedge weight vector; defaults to all ones.
%
% Outputs:
%   Theta_conv : (N x N) normalized propagation matrix
%   D_v        : (N x N) diagonal node-degree matrix
%   D_e        : (E x E) diagonal hyperedge-degree matrix
%   W          : (E x E) diagonal hyperedge-weight matrix

[num_nodes, num_edges] = size(H);

% -----------------------------------------------------------------------
% 1. Hyperedge weight matrix W
% -----------------------------------------------------------------------
if nargin < 2 || isempty(edge_weights)
    edge_weights = ones(num_edges, 1);   % Default: uniform weights
end
edge_weights = edge_weights(:);
if numel(edge_weights) ~= num_edges
    error('compute_propagation_matrix: edge_weights length must match the number of hyperedges.');
end
W = spdiags(edge_weights, 0, num_edges, num_edges);     % (E x E)

% -----------------------------------------------------------------------
% 2. Node-degree matrix D_v
%    d_v(i) = sum_e W(e,e) * H(i,e), including hyperedge weights.
% -----------------------------------------------------------------------
d_v = H * edge_weights;                  % (N x 1)
D_v = spdiags(d_v, 0, num_nodes, num_nodes);            % (N x N)

% -----------------------------------------------------------------------
% 3. Hyperedge-degree matrix D_e
%    d_e(e) = number of nodes that belong to hyperedge e.
% -----------------------------------------------------------------------
d_e = sum(H, 1)';                        % (E x 1)
D_e = spdiags(d_e, 0, num_edges, num_edges);            % (E x E)

% -----------------------------------------------------------------------
% 4. Compute D_v^(-1/2) and D_e^(-1), including zero-degree handling
% -----------------------------------------------------------------------
dv_invsqrt = d_v .^ (-0.5);
dv_invsqrt(isinf(dv_invsqrt)) = 0;      % Handle degree-zero nodes
Dv_invsqrt = spdiags(dv_invsqrt, 0, num_nodes, num_nodes);  % (N x N)

de_inv = d_e .^ (-1);
de_inv(isinf(de_inv)) = 0;              % Handle empty hyperedges
De_inv = spdiags(de_inv, 0, num_edges, num_edges);      % (E x E)

% -----------------------------------------------------------------------
% 5. Assemble the normalized propagation matrix
%    Theta_conv = Dv^(-1/2) * H * W * De^(-1) * H' * Dv^(-1/2)
% -----------------------------------------------------------------------
Theta_conv = Dv_invsqrt * H * W * De_inv * H' * Dv_invsqrt;

end
