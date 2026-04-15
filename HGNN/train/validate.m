function [accuracy, pred_labels] = validate(Y_pred, Y_true, mask)
% VALIDATE  evaluate와 동일한 평가 래퍼입니다.

if nargin < 3
    mask = [];
end

[accuracy, pred_labels] = evaluate(Y_pred, Y_true, mask);

end
