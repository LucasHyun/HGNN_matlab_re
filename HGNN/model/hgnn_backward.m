function grads = hgnn_backward(dZ2, caches, Theta_conv, weights)
% HGNN_BACKWARD  2-layer HGNN backward pass
%
% Inputs:
%   dZ2        : (N x C)      softmax + cross-entropy gradient
%   caches     : layer caches saved during the forward pass
%   Theta_conv : (N x N)      normalized propagation matrix
%   weights    : struct (.W1, .b1, .W2, .b2)
%
% Outputs:
%   grads : struct (.W1, .b1, .W2, .b2)

cache1 = caches{1};
cache2 = caches{2};

% Layer 2: Z2 = Theta_conv * (H1 * W2 + b2)
dLinear2 = Theta_conv' * dZ2;
grads.W2 = cache2.X' * dLinear2;
grads.b2 = sum(dLinear2, 1);
dH1 = dLinear2 * weights.W2';

% Undo hidden-layer dropout before applying the ReLU derivative.
if isfield(cache1, 'dropout_mask') && ~isempty(cache1.dropout_mask)
    dH1 = (dH1 .* cache1.dropout_mask) / cache1.dropout_keep_prob;
end

% Layer 1: H1 = ReLU(Z1), Z1 = Theta_conv * (X * W1 + b1)
dZ1 = dH1 .* (cache1.Z > 0);
dLinear1 = Theta_conv' * dZ1;
grads.W1 = cache1.X' * dLinear1;
grads.b1 = sum(dLinear1, 1);

end
