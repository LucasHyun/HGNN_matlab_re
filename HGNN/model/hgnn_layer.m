function [X_out, cache] = hgnn_layer(X, Theta_conv, W_layer, activation)
% HGNN_LAYER  Compute one HGNN layer
%
% Equation:
%   X_out = activation( Theta_conv * X * W_layer )
%
% Inputs:
%   X          : (N x F_in) node-feature matrix
%   Theta_conv : (N x N) normalized propagation matrix from compute_laplacian
%   W_layer    : (F_in x F_out) trainable weight matrix
%   activation : activation name ('relu' | 'softmax' | 'none')
%                Defaults to 'relu' when omitted.
%
% Outputs:
%   X_out : (N x F_out) output node-feature matrix
%   cache : intermediate values needed for backpropagation

if nargin < 4
    activation = 'relu';
end

% -----------------------------------------------------------------------
% 1. Graph propagation: aggregate neighborhood information
%    AX = Theta_conv * X   ->  (N x F_in)
% -----------------------------------------------------------------------
AX = Theta_conv * X;

% -----------------------------------------------------------------------
% 2. Linear transform: change the feature dimension
%    Z = AX * W_layer       ->  (N x F_out)
% -----------------------------------------------------------------------
Z = AX * W_layer;

% -----------------------------------------------------------------------
% 3. Apply the activation function
% -----------------------------------------------------------------------
switch lower(activation)
    case 'relu'
        X_out = max(0, Z);

    case 'softmax'
        % Subtract the row-wise max for numerical stability.
        Z_shifted = bsxfun(@minus, Z, max(Z, [], 2));
        expZ = exp(Z_shifted);
        X_out = bsxfun(@rdivide, expZ, sum(expZ, 2));

    case 'none'
        X_out = Z;

    otherwise
        error('hgnn_layer: unsupported activation "%s"', activation);
end

% -----------------------------------------------------------------------
% 4. Store cache values for backpropagation
% -----------------------------------------------------------------------
cache.X          = X;
cache.AX         = AX;
cache.Z          = Z;
cache.X_out      = X_out;
cache.W_layer    = W_layer;
cache.Theta_conv = Theta_conv;
cache.activation = activation;

end
