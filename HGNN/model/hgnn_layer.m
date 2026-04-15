function [X_out, cache] = hgnn_layer(X, Theta_conv, W_layer, activation)
% HGNN_LAYER  단일 HGNN 레이어 연산
%
% 수식:
%   X_out = activation( Theta_conv * X * W_layer )
%
% 입력:
%   X          : (N x F_in)  입력 노드 피처 행렬
%   Theta_conv : (N x N)     정규화 전파 행렬 (compute_laplacian 출력)
%   W_layer    : (F_in x F_out) 학습 가능한 가중치 행렬
%   activation : 활성화 함수 문자열 ('relu' | 'softmax' | 'none')
%                (생략 시 'relu')
%
% 출력:
%   X_out : (N x F_out) 출력 노드 피처 행렬
%   cache : 역전파에 필요한 중간값 저장 구조체

if nargin < 4
    activation = 'relu';
end

% -----------------------------------------------------------------------
% 1. 그래프 전파: 이웃 정보 집계
%    AX = Theta_conv * X   →  (N x F_in)
% -----------------------------------------------------------------------
AX = Theta_conv * X;

% -----------------------------------------------------------------------
% 2. 선형 변환: 피처 차원 변환
%    Z = AX * W_layer       →  (N x F_out)
% -----------------------------------------------------------------------
Z = AX * W_layer;

% -----------------------------------------------------------------------
% 3. 활성화 함수 적용
% -----------------------------------------------------------------------
switch lower(activation)
    case 'relu'
        X_out = max(0, Z);

    case 'softmax'
        % 수치 안정성을 위해 max 빼기
        Z_shifted = bsxfun(@minus, Z, max(Z, [], 2));
        expZ = exp(Z_shifted);
        X_out = bsxfun(@rdivide, expZ, sum(expZ, 2));

    case 'none'
        X_out = Z;

    otherwise
        error('hgnn_layer: 지원하지 않는 활성화 함수 "%s"', activation);
end

% -----------------------------------------------------------------------
% 4. 역전파용 cache 저장
% -----------------------------------------------------------------------
cache.X          = X;
cache.AX         = AX;
cache.Z          = Z;
cache.X_out      = X_out;
cache.W_layer    = W_layer;
cache.Theta_conv = Theta_conv;
cache.activation = activation;

end
