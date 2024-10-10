clc; close all; clear; rng("default"); warning("off","all"); format short

% Data set
[M, A, B] = bank_marketing(); % Load the Bank Marketing dataset
% [M, A, B] = default_credit(); % Load the Default of Credit Card dataset
% [M, A, B] = crop_mapping(); % Load the Crop Mapping dataset
% [M, A, B] = LFW(); % Load the LFW dataset.

%  If you encounter an error when opening the LFW dataset, 
% ensure it is placed inside the "images" folder and then rerun the script.

% Set the desired reduced dimensions range
r_start = 1;
r_end = 10;

% Total dimensions
r_total = r_end - r_start + 1;

% Extract the dimension
n = size(A, 2);
m1 = size(A, 1);
m2 = size(B, 1);
m = m1 + m2;

% Parameters
tol = 1e-8; % Tolerance for optimization

% Initialize runtime arrays
time_pca = zeros(r_total, 1);
time_Fair = zeros(r_total, 1);
time_CFPCA = zeros(r_total, 1);
time_ratio1 = zeros(r_total, 1);
time_ratio2 = zeros(r_total, 1);

for idx = 1:r_total
    % PCA 
    r = r_start + idx - 1;

    tic;
    coeff = pca(M, "NumComponents", r); % Perform PCA
    time_pca(idx) = toc;

    % Projection of PCA
    approx_Mpca = M * (coeff * coeff');
    approx_Apca = A * (coeff * coeff');
    approx_Bpca = B * (coeff * coeff');

    % FPCAviaEigOpt
    tic;
    [U] = FPCAviaEigOpt(A, B, r, tol); % Perform FPCAviaEigOpt
    time_Fair(idx) = toc;

    % Projection of FPCAviaEigOpt
    approx_A = A * (U * U');
    approx_B = B * (U * U');

    % c_FPCA
    tic;
    P_CFPCA = c_FPCA(A, B, r, tol); % Perform CFPCA
    time_CFPCA(idx) = toc;

    % Projection of CFPCA
    approxFair_M_CFPCA = M * P_CFPCA;
    approxFair_A_CFPCA = A * P_CFPCA;
    approxFair_B_CFPCA = B * P_CFPCA;


    % Runtime ratio between FPCAviaEigOpt and CFPCA
    time_ratio1(idx) = time_Fair(idx) / time_pca(idx);
    time_ratio2(idx) = time_CFPCA(idx) / time_Fair(idx);

end

% Reduced dimension count
r_count = (r_start:r_end)';

T = table(r_count, time_pca, time_Fair, time_CFPCA, time_ratio1, time_ratio2,...
    'VariableNames', {'r', 'PCA', 'FPCAviaEigOpt', 'CFPCA', 'Time_Ratio1','Time_Ratio2'});
disp(T)
