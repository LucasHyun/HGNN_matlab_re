function [accuracy, pred_labels] = evaluate(Y_pred, Y_true, mask)
% EVALUATE  Evaluate node-classification accuracy
%
% Inputs:
%   Y_pred : (N x C) softmax output with class probabilities
%   Y_true : (N x C) one-hot ground-truth labels
%   mask   : (N x 1) node mask to evaluate; uses all nodes when omitted.
%
% Outputs:
%   accuracy    : scalar accuracy in the range [0, 1]
%   pred_labels : (N x 1) predicted class indices

if nargin < 3 || isempty(mask)
    mask = true(size(Y_pred, 1), 1);
end
mask = logical(mask(:));

num_eval = sum(mask);
if num_eval == 0
    error('evaluate: the mask does not include any nodes.');
end

[~, pred_labels] = max(Y_pred, [], 2);
[~, true_labels] = max(Y_true, [], 2);

correct = pred_labels(mask) == true_labels(mask);
accuracy = sum(correct) / num_eval;

end
