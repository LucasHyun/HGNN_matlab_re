function [Y_pred, caches] = hgnn_forward(X, Theta_conv, weights)
% HGNN_FORWARD  Full HGNN forward pass for the two-layer model
%
% Structure:
%   Layer 1: ReLU( Theta_conv * X        * W1 )  ->  hidden
%   Layer 2: Softmax( Theta_conv * hidden * W2 )  ->  Y_pred
%
% Inputs:
%   X          : (N x F_in) input node features
%   Theta_conv : (N x N) normalized propagation matrix
%   weights    : struct
%       .W1    : (F_in  x F_hidden) first-layer weights
%       .W2    : (F_hidden x F_out) second-layer weights
%       Add .W3 or more fields when extending the architecture.
%
% Outputs:
%   Y_pred : (N x F_out) class probabilities from softmax
%   caches : cell array with one backpropagation cache per layer

caches = cell(2, 1);

% -----------------------------------------------------------------------
% Layer 1: input -> hidden features (ReLU)
% -----------------------------------------------------------------------
[H1, caches{1}] = hgnn_layer(X, Theta_conv, weights.W1, 'relu');

% -----------------------------------------------------------------------
% Optional dropout for training only.
%   H1 = dropout(H1, dropout_rate);   % TODO: implement if needed.
% -----------------------------------------------------------------------

% -----------------------------------------------------------------------
% Layer 2: hidden features -> output probabilities (Softmax)
% -----------------------------------------------------------------------
[Y_pred, caches{2}] = hgnn_layer(H1, Theta_conv, weights.W2, 'softmax');

end
