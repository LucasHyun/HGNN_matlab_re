function [accuracy, pred_labels] = evaluate(Y_pred, Y_true, mask)
% EVALUATE  노드 분류 정확도 평가
%
% 입력:
%   Y_pred : (N x C)  softmax 출력 (클래스 확률)
%   Y_true : (N x C)  one-hot 정답 레이블
%   mask   : (N x 1)  평가할 노드 마스크 (생략 시 전체)
%
% 출력:
%   accuracy    : 스칼라, 정확도 (0~1)
%   pred_labels : (N x 1) 예측 클래스 인덱스

if nargin < 3 || isempty(mask)
    mask = true(size(Y_pred, 1), 1);
end
mask = logical(mask(:));

num_eval = sum(mask);
if num_eval == 0
    error('evaluate: mask에 포함된 노드가 없습니다.');
end

[~, pred_labels] = max(Y_pred, [], 2);
[~, true_labels] = max(Y_true, [], 2);

correct = pred_labels(mask) == true_labels(mask);
accuracy = sum(correct) / num_eval;

end
