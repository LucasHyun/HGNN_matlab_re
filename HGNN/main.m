% MAIN.M  Run the complete HGNN pipeline
% =========================================================================
%  Execution order:
%   1. Load the dataset
%   2. Build the incidence matrix and normalized propagation matrix
%   3. Train the HGNN
%   4. Evaluate on the test split
% =========================================================================

clc; clear; close all;

project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'data'), ...
        fullfile(project_root, 'graph'), ...
        fullfile(project_root, 'model'), ...
        fullfile(project_root, 'train'));

%% -------------------------------------------------------------------------
% 1. Load data
% -------------------------------------------------------------------------
dataset = 'cora';  % 'toy' | 'cora' | 'custom'

% Cora hyperedge construction options:
% - Use papers cited at least min_group_citations times as research-group seeds.
% - Each hyperedge contains the seed paper and the papers that cite it.
data_options.min_group_citations = 5;
data_options.include_singletons  = true;

[X, H, Y_true, train_mask, val_mask, test_mask] = load_data(dataset, data_options);

fprintf('=== Data loaded ===\n');
fprintf('  Nodes      : %d\n', size(X, 1));
fprintf('  Features   : %d\n', size(X, 2));
fprintf('  Hyperedges : %d\n', size(H, 2));
fprintf('  Classes    : %d\n', size(Y_true, 2));
fprintf('  Train/Val/Test : %d / %d / %d\n', ...
        sum(train_mask), sum(val_mask), sum(test_mask));

%% -------------------------------------------------------------------------
% 2. Compute the normalized propagation matrix
% -------------------------------------------------------------------------
[Theta_conv, D_v, D_e, W] = compute_laplacian(H);

fprintf('\n=== Propagation matrix computed ===\n');

%% -------------------------------------------------------------------------
% 3. Configure hyperparameters
% -------------------------------------------------------------------------
params.lr           = 0.01;
params.epochs       = 200;
params.hidden_dim   = 64;
params.weight_decay = 5e-4;
params.print_every  = 10;

%% -------------------------------------------------------------------------
% 4. Train
% -------------------------------------------------------------------------
fprintf('\n=== Training started ===\n');
weights = train(X, Theta_conv, Y_true, train_mask, val_mask, params);
fprintf('=== Training completed ===\n');

%% -------------------------------------------------------------------------
% 5. Evaluate on the test split
% -------------------------------------------------------------------------
[Y_pred, ~] = hgnn_forward(X, Theta_conv, weights);
[test_acc, ~] = evaluate(Y_pred, Y_true, test_mask);

fprintf('\n=== Final test accuracy: %.4f ===\n', test_acc);
