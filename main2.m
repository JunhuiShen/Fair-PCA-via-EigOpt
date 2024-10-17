clc; close all; clear; rng("default"); warning("off", "all"); format short

% Dataset
% [M, A, B] = bank_marketing();   % Bank Marketing dataset
% [M, A, B] = default_credit(); % Default of Credit Card dataset
% [M, A, B] = crop_mapping();   % Crop Mapping dataset
[M, A, B] = LFW();            % LFW dataset

% Set the range for reduced dimensions
r_start = 200;   
r_end = 200;    
r_total = r_end - r_start + 1; 

% Parameters for optimization methods
tol = 1e-8;  

% Initialize reconstruction loss arrays
rloss_A = zeros(r_total, 1);
rloss_B = zeros(r_total, 1);
rlossFair_A = zeros(r_total, 1);
rlossFair_B = zeros(r_total, 1);
rloss_A_UFPCA = zeros(r_total, 1);
rloss_B_UFPCA = zeros(r_total, 1);
rloss_A_CFPCA = zeros(r_total, 1);
rloss_B_CFPCA = zeros(r_total, 1);

% Initialize arrays to store runtime results for each method
time_pca = zeros(r_total, 1);    
time_Fair = zeros(r_total, 1);   
time_CFPCA = zeros(r_total, 1);  
time_UFPCA = zeros(r_total, 1);  


for idx = 1:r_total

    r = r_start + idx - 1;

    % PCA
    tic;  
    coeff = pca(M, "NumComponents", r); 
    time_pca(idx) = toc; 

    % Projection of PCA
    approx_Mpca = M * (coeff * coeff');
    approx_Apca = A * (coeff * coeff');
    approx_Bpca = B * (coeff * coeff');

    % Reconstruction loss for PCA
    rloss_A(idx) = rloss(A, approx_Apca, r); % rloss is already divided by the sample size
    rloss_B(idx) = rloss(B, approx_Bpca, r);

    % FPCA via Eigenvalue Optimization (FPCA via EigOpt)
    tic;  
    [U] = FPCAviaEigOpt(A, B, r, tol);  
    time_Fair(idx) = toc; 

    % Projection of FPCAviaEigOpt
    approx_A = A * (U * U');
    approx_B = B * (U * U');

    % Reconstruction loss for FPCAviaEigOpt
    rlossFair_A(idx) = rloss(A, approx_A, r);
    rlossFair_B(idx) = rloss(B, approx_B, r);

        
    % CFPCA (Unconstrained Fair PCA)
    tic;  
    P_UFPCA = u_FPCA(A, B, r, tol);  
    time_UFPCA(idx) = toc; 

    % Projection of CFPCA
    approxFair_A_UFPA = A * P_UFPCA;
    approxFair_B_UFPCA = B * P_UFPCA;

    % Reconstruction loss for CFPCA
    rloss_A_UFPCA(idx) = rloss(A, approxFair_A_UFPA, r); 
    rloss_B_UFPCA(idx) = rloss(B, approxFair_B_UFPCA, r); 

    % CFPCA (Constrained Fair PCA)
    tic;  
    P_CFPCA = c_FPCA(A, B, r, tol);  
    time_CFPCA(idx) = toc; 

    % Projection of CFPCA
    approxFair_A_CFPA = A * P_CFPCA;
    approxFair_B_CFPCA = B * P_CFPCA;

    % Reconstruction loss for CFPCA
    rloss_A_CFPCA(idx) = rloss(A, approxFair_A_CFPA, r); 
    rloss_B_CFPCA(idx) = rloss(B, approxFair_B_CFPCA, r); 
end

% Runtime table
r_count = (r_start:r_end)';  
T1 = table(r_count, time_pca, time_Fair, time_UFPCA,  time_CFPCA,...
    'VariableNames', {'r', 'PCA', 'FPCAviaEigOpt', 'UFPCA', 'CFPCA'});
disp(T1)

% LossError table
T2 = table(r_count, abs(1-rloss_A./rloss_B), abs(1 - rlossFair_A ./ rlossFair_B), ...
    abs(1 - rloss_A_UFPCA ./ rloss_B_UFPCA),  abs(1 - rloss_A_CFPCA ./ rloss_B_CFPCA),...
    'VariableNames', {'r', 'Rloss_err_PCA', 'LossRatio_err_EigOpt', 'LossRatio_err_UFPCA','LossRatio_err_CFPCA'});
disp(T2)