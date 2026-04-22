function [loss, dY] = hgnn_loss(Y_pred, Y_true, mask)
% HGNN_LOSS  Compute cross-entropy loss
%
% Inputs:
%   Y_pred : (N x C) softmax output with class probabilities
%   Y_true : (N x C) one-hot ground-truth labels
%   mask   : (N x 1) node mask used for training (1=include, 0=exclude)
%            Uses all nodes when omitted.
%
% Outputs:
%   loss : scalar mean cross-entropy loss
%   dY   : (N x C) combined softmax + cross-entropy gradient

if nargin < 3 || isempty(mask)
    mask = true(size(Y_pred, 1), 1);
end
mask = logical(mask(:));

num_samples = sum(mask);
if num_samples == 0
    error('hgnn_loss: the mask does not include any training nodes.');
end

epsilon = 1e-8;   % Avoid log(0)

% -----------------------------------------------------------------------
% 1. Compute cross-entropy loss
%    loss = -1/N * sum_i sum_c [ Y_true(i,c) * log(Y_pred(i,c)) ]
%    Only masked nodes contribute to the loss.
% -----------------------------------------------------------------------
log_pred = log(Y_pred + epsilon);                      % (N x C)
ce_per_node = -sum(Y_true .* log_pred, 2);             % (N x 1)
loss = sum(ce_per_node .* mask) / num_samples;         % scalar

% -----------------------------------------------------------------------
% 2. Compute the gradient
%    Combined softmax + cross-entropy gradient:
%    dY = (Y_pred - Y_true) / N, with the mask applied.
% -----------------------------------------------------------------------
dY = bsxfun(@times, Y_pred - Y_true, double(mask)) / num_samples;  % (N x C)

end
