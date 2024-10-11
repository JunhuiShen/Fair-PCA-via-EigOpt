clc; close all; clear; rng("default"); warning("off", "all"); format short

% Dataset
[M, A, B] = bank_marketing();   % Bank Marketing dataset
% [M, A, B] = default_credit(); % Default of Credit Card dataset
% [M, A, B] = crop_mapping();   % Crop Mapping dataset
% [M, A, B] = LFW();            % LFW dataset

% Set the range for reduced dimensions
r_start = 1;   % Starting reduced dimension
r_end = 10;    % Ending reduced dimension
r_total = r_end - r_start + 1;  % Total number of reduced dimensions

% Parameters for optimization methods
tol = 1e-8;  % Tolerance for optimization convergence

% Initialize arrays to store runtime results for each method
time_pca = zeros(r_total, 1);    % Runtime for PCA
time_Fair = zeros(r_total, 1);   % Runtime for FPCA via EigOpt
time_CFPCA = zeros(r_total, 1);  % Runtime for CFPCA

% Loop through each reduced dimension r in the range
for idx = 1:r_total
    % Set current reduced dimension
    r = r_start + idx - 1;

    % PCA
    tic;  % Start timer for PCA
    coeff = pca(M, "NumComponents", r); % Perform PCA with r components
    time_pca(idx) = toc;  % Store PCA runtime

    % FPCA via Eigenvalue Optimization (FPCA via EigOpt)
    tic;  % Start timer for FPCA via EigOpt
    [U] = FPCAviaEigOpt(A, B, r, tol);  % Perform FPCA via EigOpt
    time_Fair(idx) = toc;  % Store FPCA via EigOpt runtime

    % CFPCA (Constrained Fair PCA)
    tic;  % Start timer for CFPCA
    P_CFPCA = c_FPCA(A, B, r, tol);  % Perform CFPCA
    time_CFPCA(idx) = toc;  % Store CFPCA runtime
end

% Create a table summarizing the runtimes for PCA, FPCA via EigOpt, and CFPCA
r_count = (r_start:r_end)';  % Array of reduced dimensions
T = table(r_count, time_pca, time_Fair, time_CFPCA, ...
    'VariableNames', {'r', 'PCA', 'FPCAviaEigOpt', 'CFPCA'});

% Display the table
disp(T);

