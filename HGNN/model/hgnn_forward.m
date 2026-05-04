function [Y_pred, caches] = hgnn_forward(X, Theta_conv, weights, is_training, params)
% HGNN_FORWARD  Full HGNN forward pass for the two-layer model
%
% Structure:
%   Layer 1: ReLU( Theta_conv * (X * W1 + b1) )  ->  hidden
%   Dropout: applied to the hidden representation during training only
%   Layer 2: Softmax( Theta_conv * (hidden * W2 + b2) )  ->  Y_pred
%
% Inputs:
%   X          : (N x F_in) input node features
%   Theta_conv : (N x N) normalized propagation matrix
%   weights    : struct
%       .W1    : (F_in  x F_hidden) first-layer weights
%       .b1    : (1 x F_hidden) first-layer bias
%       .W2    : (F_hidden x F_out) second-layer weights
%       .b2    : (1 x F_out) second-layer bias
%       Add .W3 or more fields when extending the architecture.
%   is_training: logical flag. Dropout is active only when this is true.
%   params     : optional struct
%       .dropout : hidden-layer dropout probability (default 0)
%
% Outputs:
%   Y_pred : (N x F_out) class probabilities from softmax
%   caches : cell array with one backpropagation cache per layer

if nargin < 4 || isempty(is_training)
    is_training = false;
end
if nargin < 5 || isempty(params)
    params = struct();
end
dropout = getfield_default(params, 'dropout', 0);

caches = cell(2, 1);

% -----------------------------------------------------------------------
% Layer 1: input -> hidden features (ReLU)
% -----------------------------------------------------------------------
[H1, caches{1}] = hgnn_layer(X, Theta_conv, weights.W1, weights.b1, 'relu');

% -----------------------------------------------------------------------
% Dropout: match the original HGNN training recipe while keeping inference
% deterministic. Inverted dropout keeps the expected activation scale.
% -----------------------------------------------------------------------
if is_training && dropout > 0
    if dropout >= 1
        error('hgnn_forward: dropout must be smaller than 1.');
    end
    keep_prob = 1 - dropout;
    dropout_mask = rand(size(H1)) < keep_prob;
    H1 = (H1 .* dropout_mask) / keep_prob;
    caches{1}.dropout_mask = dropout_mask;
    caches{1}.dropout_keep_prob = keep_prob;
else
    caches{1}.dropout_mask = [];
    caches{1}.dropout_keep_prob = 1;
end

% -----------------------------------------------------------------------
% Layer 2: hidden features -> output probabilities (Softmax)
% -----------------------------------------------------------------------
[Y_pred, caches{2}] = hgnn_layer(H1, Theta_conv, weights.W2, weights.b2, 'softmax');

end

function val = getfield_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
