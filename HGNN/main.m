% MAIN.M  HGNN 전체 파이프라인 실행
% =========================================================================
%  실행 순서:
%   1. 데이터 로드
%   2. Incidence matrix & Laplacian 계산
%   3. 학습
%   4. 테스트 평가
% =========================================================================

clc; clear; close all;

project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'data'), ...
        fullfile(project_root, 'graph'), ...
        fullfile(project_root, 'model'), ...
        fullfile(project_root, 'train'));

%% -------------------------------------------------------------------------
% 1. 데이터 로드
% -------------------------------------------------------------------------
dataset = 'cora';  % 'toy' | 'cora' | 'custom'
[X, H, Y_true, train_mask, val_mask, test_mask] = load_data(dataset);

fprintf('=== 데이터 로드 완료 ===\n');
fprintf('  노드 수    : %d\n', size(X, 1));
fprintf('  피처 차원  : %d\n', size(X, 2));
fprintf('  하이퍼엣지 : %d\n', size(H, 2));
fprintf('  클래스 수  : %d\n', size(Y_true, 2));
fprintf('  Train/Val/Test : %d / %d / %d\n', ...
        sum(train_mask), sum(val_mask), sum(test_mask));

%% -------------------------------------------------------------------------
% 2. 정규화 전파 행렬 계산
% -------------------------------------------------------------------------
[Theta_conv, D_v, D_e, W] = compute_laplacian(H);

fprintf('\n=== Laplacian 계산 완료 ===\n');

%% -------------------------------------------------------------------------
% 3. 하이퍼파라미터 설정
% -------------------------------------------------------------------------
params.lr           = 0.01;
params.epochs       = 200;
params.hidden_dim   = 64;
params.weight_decay = 5e-4;
params.print_every  = 10;

%% -------------------------------------------------------------------------
% 4. 학습
% -------------------------------------------------------------------------
fprintf('\n=== 학습 시작 ===\n');
weights = train(X, Theta_conv, Y_true, train_mask, val_mask, params);
fprintf('=== 학습 완료 ===\n');

%% -------------------------------------------------------------------------
% 5. 테스트 평가
% -------------------------------------------------------------------------
[Y_pred, ~] = hgnn_forward(X, Theta_conv, weights);
[test_acc, ~] = evaluate(Y_pred, Y_true, test_mask);

fprintf('\n=== 최종 테스트 정확도: %.4f ===\n', test_acc);
