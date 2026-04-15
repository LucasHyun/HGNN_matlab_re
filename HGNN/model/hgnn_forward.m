function [Y_pred, caches] = hgnn_forward(X, Theta_conv, weights)
% HGNN_FORWARD  전체 HGNN Forward Pass (2레이어 구조)
%
% 구조:
%   Layer 1: ReLU( Theta_conv * X        * W1 )  →  hidden
%   Layer 2: Softmax( Theta_conv * hidden * W2 )  →  Y_pred
%
% 입력:
%   X          : (N x F_in)   입력 노드 피처
%   Theta_conv : (N x N)      정규화 전파 행렬
%   weights    : 구조체
%       .W1    : (F_in  x F_hidden)  1번째 레이어 가중치
%       .W2    : (F_hidden x F_out)  2번째 레이어 가중치
%       (확장 시 .W3 추가 가능)
%
% 출력:
%   Y_pred : (N x F_out)  클래스별 확률 (softmax 출력)
%   caches : cell array, 각 레이어의 cache 저장 (역전파용)

caches = cell(2, 1);

% -----------------------------------------------------------------------
% Layer 1: 입력 → 은닉층  (ReLU)
% -----------------------------------------------------------------------
[H1, caches{1}] = hgnn_layer(X, Theta_conv, weights.W1, 'relu');

% -----------------------------------------------------------------------
% (선택) Dropout - 학습 시에만 적용
%   H1 = dropout(H1, dropout_rate);   % TODO: 필요 시 구현
% -----------------------------------------------------------------------

% -----------------------------------------------------------------------
% Layer 2: 은닉층 → 출력층  (Softmax)
% -----------------------------------------------------------------------
[Y_pred, caches{2}] = hgnn_layer(H1, Theta_conv, weights.W2, 'softmax');

end