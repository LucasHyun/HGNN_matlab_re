function [loss, dY] = hgnn_loss(Y_pred, Y_true, mask)
% HGNN_LOSS  Cross-Entropy Loss 계산
%
% 입력:
%   Y_pred : (N x C)  softmax 출력 (클래스 확률)
%   Y_true : (N x C)  one-hot 정답 레이블
%   mask   : (N x 1)  학습에 사용할 노드 마스크 (1=학습, 0=제외)
%            (생략 시 전체 노드 사용)
%
% 출력:
%   loss : 스칼라, 평균 cross-entropy loss
%   dY   : (N x C) softmax + cross-entropy 결합 gradient

if nargin < 3 || isempty(mask)
    mask = true(size(Y_pred, 1), 1);
end
mask = logical(mask(:));

num_samples = sum(mask);
if num_samples == 0
    error('hgnn_loss: mask에 포함된 학습 노드가 없습니다.');
end

epsilon = 1e-8;   % log(0) 방지

% -----------------------------------------------------------------------
% 1. Cross-Entropy Loss 계산
%    loss = -1/N * sum_i sum_c [ Y_true(i,c) * log(Y_pred(i,c)) ]
%    (mask된 노드만 포함)
% -----------------------------------------------------------------------
log_pred = log(Y_pred + epsilon);                      % (N x C)
ce_per_node = -sum(Y_true .* log_pred, 2);             % (N x 1)
loss = sum(ce_per_node .* mask) / num_samples;         % 스칼라

% -----------------------------------------------------------------------
% 2. Gradient 계산
%    softmax + cross-entropy 결합 gradient:
%    dY = (Y_pred - Y_true) / N   (mask 적용)
% -----------------------------------------------------------------------
dY = bsxfun(@times, Y_pred - Y_true, double(mask)) / num_samples;  % (N x C)

end
