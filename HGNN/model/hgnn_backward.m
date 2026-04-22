function grads = hgnn_backward(dZ2, caches, Theta_conv, weights)
% HGNN_BACKWARD  2-layer HGNN backward pass
%
% Inputs:
%   dZ2        : (N x C)      softmax + cross-entropy gradient
%   caches     : layer caches saved during the forward pass
%   Theta_conv : (N x N)      normalized propagation matrix
%   weights    : struct (.W1, .W2)
%
% Outputs:
%   grads : struct (.W1, .W2)

cache1 = caches{1};
cache2 = caches{2};

% Layer 2: Z2 = Theta_conv * H1 * W2
grads.W2 = cache2.AX' * dZ2;
dAX2 = dZ2 * weights.W2';
dH1 = Theta_conv' * dAX2;

% Layer 1: H1 = ReLU(Z1), Z1 = Theta_conv * X * W1
dZ1 = dH1 .* (cache1.Z > 0);
grads.W1 = cache1.AX' * dZ1;

end
