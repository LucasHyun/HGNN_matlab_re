function [accuracy, pred_labels] = validate(Y_pred, Y_true, mask)
% VALIDATE  Thin evaluation wrapper around evaluate.

if nargin < 3
    mask = [];
end

[accuracy, pred_labels] = evaluate(Y_pred, Y_true, mask);

end
