function weights = train(X, Theta_conv, Y_true, train_mask, val_mask, params)
% TRAIN  HGNN 학습 루프
%
% 입력:
%   X          : (N x F_in)   node-feature
%   Theta_conv : (N x N)      regularized propagating matrix
%   Y_true     : (N x C)      one-hot label
%   train_mask : (N x 1)      학습 노드 마스크
%   val_mask   : (N x 1)      검증 노드 마스크
%   params     : 하이퍼파라미터 구조체
%       .lr          : 학습률 (기본 0.01)
%       .epochs      : 에폭 수 (기본 200)
%       .hidden_dim  : 은닉층 차원 (기본 64)
%       .weight_decay: L2 정규화 계수 (기본 5e-4)
%
% 출력:
%   weights : 학습된 가중치 구조체 (.W1, .W2)

% -----------------------------------------------------------------------
% 0. 하이퍼파라미터 기본값 설정
% -----------------------------------------------------------------------
if nargin < 6, params = struct(); end
lr           = getfield_default(params, 'lr',           0.01);
epochs       = getfield_default(params, 'epochs',       200);
hidden_dim   = getfield_default(params, 'hidden_dim',   64);
weight_decay = getfield_default(params, 'weight_decay', 5e-4);
print_every  = getfield_default(params, 'print_every',  10);

[~, F_in] = size(X);
[~, C]    = size(Y_true);

% -----------------------------------------------------------------------
% 1. 가중치 초기화 (Xavier)
% -----------------------------------------------------------------------
weights.W1 = xavier_init(F_in, hidden_dim);
weights.W2 = xavier_init(hidden_dim, C);

% Adam optimizer 상태 초기화 (TODO: 필요 시 SGD와 선택 구조)
adam = init_adam(weights);

% -----------------------------------------------------------------------
% 2. 학습 루프
% -----------------------------------------------------------------------
for epoch = 1:epochs

    % --- Forward pass ---
    [Y_pred, caches] = hgnn_forward(X, Theta_conv, weights);

    % --- Loss 계산 (train 노드만) ---
    [loss, dY] = hgnn_loss(Y_pred, Y_true, train_mask);

    % --- Backward pass (gradient 계산) ---
    grads = hgnn_backward(dY, caches, Theta_conv, weights);

    % --- L2 정규화 gradient 추가 ---
    grads.W1 = grads.W1 + weight_decay * weights.W1;
    grads.W2 = grads.W2 + weight_decay * weights.W2;

    % --- 가중치 업데이트 (Adam) ---
    [weights, adam] = adam_update(weights, grads, adam, lr, epoch);

    % --- 검증 정확도 출력 ---
    if mod(epoch, print_every) == 0
        [Y_val_pred, ~] = hgnn_forward(X, Theta_conv, weights);
        train_acc = evaluate(Y_val_pred, Y_true, train_mask);
        val_acc = evaluate(Y_val_pred, Y_true, val_mask);
        fprintf('Epoch %3d | Loss: %.4f | Train Acc: %.4f | Val Acc: %.4f\n', ...
                epoch, loss, train_acc, val_acc);
    end
end

end

% -----------------------------------------------------------------------
% 내부 헬퍼 함수
% -----------------------------------------------------------------------
function W = xavier_init(fan_in, fan_out)
    scale = sqrt(2.0 / (fan_in + fan_out));
    W = randn(fan_in, fan_out) * scale;
end

function adam = init_adam(weights)
    adam.m.W1 = zeros(size(weights.W1));
    adam.m.W2 = zeros(size(weights.W2));
    adam.v.W1 = zeros(size(weights.W1));
    adam.v.W2 = zeros(size(weights.W2));
    adam.beta1 = 0.9;
    adam.beta2 = 0.999;
    adam.eps   = 1e-8;
end

function [weights, adam] = adam_update(weights, grads, adam, lr, t)
    fields = {'W1', 'W2'};
    for i = 1:length(fields)
        f = fields{i};
        adam.m.(f) = adam.beta1 * adam.m.(f) + (1-adam.beta1) * grads.(f);
        adam.v.(f) = adam.beta2 * adam.v.(f) + (1-adam.beta2) * grads.(f).^2;
        m_hat = adam.m.(f) / (1 - adam.beta1^t);
        v_hat = adam.v.(f) / (1 - adam.beta2^t);
        weights.(f) = weights.(f) - lr * m_hat ./ (sqrt(v_hat) + adam.eps);
    end
end

function val = getfield_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
