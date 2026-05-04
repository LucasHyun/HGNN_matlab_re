function [weights, history] = train(X, Theta_conv, Y_true, train_mask, val_mask, params)
% TRAIN  HGNN training loop
%
% Inputs:
%   X          : (N x F_in)   node-feature
%   Theta_conv : (N x N)      normalized propagation matrix
%   Y_true     : (N x C)      one-hot label
%   train_mask : (N x 1)      training-node mask
%   val_mask   : (N x 1)      validation-node mask
%   params     : hyperparameter struct
%       .lr          : learning rate (default 0.01)
%       .epochs      : number of epochs (default 200)
%       .hidden_dim  : hidden-layer dimension (default 64)
%       .weight_decay: L2 regularization coefficient (default 5e-4)
%       .dropout     : hidden-layer dropout probability (default 0.5)
%
% Outputs:
%   weights : best validation-loss weight struct (.W1, .b1, .W2, .b2)
%   history : training/validation metrics and best epoch metadata

% -----------------------------------------------------------------------
% 0. Set default hyperparameters
% -----------------------------------------------------------------------
if nargin < 6, params = struct(); end
lr           = getfield_default(params, 'lr',           0.01);
epochs       = getfield_default(params, 'epochs',       200);
hidden_dim   = getfield_default(params, 'hidden_dim',   64);
weight_decay = getfield_default(params, 'weight_decay', 5e-4);
print_every  = getfield_default(params, 'print_every',  10);
dropout      = getfield_default(params, 'dropout',      0.5);

[~, F_in] = size(X);
[~, C]    = size(Y_true);

% -----------------------------------------------------------------------
% 1. Initialize weights with Xavier initialization
% -----------------------------------------------------------------------
weights.W1 = xavier_init(F_in, hidden_dim);
weights.b1 = zeros(1, hidden_dim);
weights.W2 = xavier_init(hidden_dim, C);
weights.b2 = zeros(1, C);

% Initialize Adam optimizer state (TODO: add SGD selection if needed).
adam = init_adam(weights);

best_weights = weights;
best_val_loss = inf;
best_epoch = 0;

history.train_loss = zeros(epochs, 1);
history.val_loss = zeros(epochs, 1);
history.train_acc = zeros(epochs, 1);
history.val_acc = zeros(epochs, 1);
history.best_epoch = best_epoch;
history.best_val_loss = best_val_loss;

% -----------------------------------------------------------------------
% 2. Training loop
% -----------------------------------------------------------------------
for epoch = 1:epochs

    % --- Forward pass ---
    train_params.dropout = dropout;
    [Y_pred, caches] = hgnn_forward(X, Theta_conv, weights, true, train_params);

    % --- Loss calculation on training nodes only ---
    [~, dY] = hgnn_loss(Y_pred, Y_true, train_mask);

    % --- Backward pass (gradient calculation) ---
    grads = hgnn_backward(dY, caches, Theta_conv, weights);

    % --- Add L2 regularization gradients ---
    grads.W1 = grads.W1 + weight_decay * weights.W1;
    grads.W2 = grads.W2 + weight_decay * weights.W2;

    % --- Weight update with Adam ---
    [weights, adam] = adam_update(weights, grads, adam, lr, epoch);

    % --- Evaluate without dropout and keep the best validation-loss model ---
    [Y_eval_pred, ~] = hgnn_forward(X, Theta_conv, weights, false, train_params);
    [train_loss, ~] = hgnn_loss(Y_eval_pred, Y_true, train_mask);
    [val_loss, ~] = hgnn_loss(Y_eval_pred, Y_true, val_mask);
    train_acc = evaluate(Y_eval_pred, Y_true, train_mask);
    val_acc = evaluate(Y_eval_pred, Y_true, val_mask);

    history.train_loss(epoch) = train_loss;
    history.val_loss(epoch) = val_loss;
    history.train_acc(epoch) = train_acc;
    history.val_acc(epoch) = val_acc;

    if val_loss < best_val_loss
        best_val_loss = val_loss;
        best_weights = weights;
        best_epoch = epoch;
    end

    % --- Print validation metrics ---
    if mod(epoch, print_every) == 0
        fprintf(['Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | ' ...
                 'Train Acc: %.4f | Val Acc: %.4f | Best Epoch: %d\n'], ...
                epoch, train_loss, val_loss, train_acc, val_acc, best_epoch);
    end
end

weights = best_weights;
history.best_epoch = best_epoch;
history.best_val_loss = best_val_loss;

end

% -----------------------------------------------------------------------
% Internal helper functions
% -----------------------------------------------------------------------
function W = xavier_init(fan_in, fan_out)
    scale = sqrt(2.0 / (fan_in + fan_out));
    W = randn(fan_in, fan_out) * scale;
end

function adam = init_adam(weights)
    fields = fieldnames(weights);
    for i = 1:length(fields)
        f = fields{i};
        adam.m.(f) = zeros(size(weights.(f)));
        adam.v.(f) = zeros(size(weights.(f)));
    end
    adam.beta1 = 0.9;
    adam.beta2 = 0.999;
    adam.eps   = 1e-8;
end

function [weights, adam] = adam_update(weights, grads, adam, lr, t)
    fields = fieldnames(weights);
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
