clc; close all; clear; rng("default"); warning("off","all"); 
format long

% Data set
[M, A, B] = bank_marketing(); % Load the Bank Marketing dataset
% [M, A, B] = default_credit(); % Load the Default of Credit Card dataset
% [M, A, B] = crop_mapping(); % Load the Crop Mapping dataset
% [M, A, B] = LFW(); % Load the LFW dataset.

%  If you encounter an error when opening the LFW dataset, 
% ensure it is placed inside the "images" folder and then rerun the script.

% Set the desired reduced dimensions range
r_start = 7;
r_end = 7;

% Total dimensions
r_total = r_end - r_start + 1;

% Extract the dimension
n = size(A, 2);
m1 = size(A, 1);
m2 = size(B, 1);
m = m1 + m2;

% Initialize reconstruction error arrays
error_M = zeros(r_total, 1);
errorFair_M = zeros(r_total, 1);

% Initialize reconstruction loss arrays
rloss_A = zeros(r_total, 1);
rloss_B = zeros(r_total, 1);
rlossFair_A = zeros(r_total, 1);
rlossFair_B = zeros(r_total, 1);
rloss_A_SDR = zeros(r_total, 1);
rloss_B_SDR = zeros(r_total, 1);

% Parameters
tol = 1e-8; % Tolerance for optimization
eta = 1; % Parameter for Fair PCA via LP
T = 10; % Number of iterations for Fair PCA via LP

% Initialize runtime arrays
time_pca = zeros(r_total, 1);
time_Fair = zeros(r_total, 1);
time_FairLP = zeros(r_total, 1);
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

    % Reconstruction error for PCA
    error_M(idx) = error1(M,approx_Mpca);

    % Reconstruction loss for PCA
    rloss_A(idx) = rloss(A, approx_Apca, r); % rloss is already divided by the sample size
    rloss_B(idx) = rloss(B, approx_Bpca, r);

    % FPCAviaEigOpt
    tic;
    [U] = FPCAviaEigOpt(A, B, r, tol); % Perform FPCAviaEigOpt
    time_Fair(idx) = toc;

    % Projection of FPCAviaEigOpt
    approx_A = A * (U * U');
    approx_B = B * (U * U');

    % Reconstruction error for FPCAviaEigOpt
    errorFair_M(idx) = error1(M,M * (U * U'));

    % Reconstruction loss for FPCAviaEigOpt
    rlossFair_A(idx) = rloss(A, approx_A, r);
    rlossFair_B(idx) = rloss(B, approx_B, r);

    % FPCAviaSDR
    tic;
    P_SDR = FPCAviaSDR(A, B, r, eta, T); % Perform FPCAviaSDR
    time_FairLP(idx) = toc;

    % Projection of FPCAviaSDR
    approxFair_M_SDR = M * P_SDR;
    approxFair_A_SDR = A * P_SDR;
    approxFair_B_SDR = B * P_SDR;

    % Reconstruction loss for FPCAviaSDR
    rloss_A_SDR(idx) = rloss(A, approxFair_A_SDR, r); 
    rloss_B_SDR(idx) = rloss(B, approxFair_B_SDR, r); 

    % Runtime ratio between FPCAviaEigOpt and FPCAviaSDR
    time_ratio1(idx) = time_Fair(idx) / time_pca(idx);
    time_ratio2(idx) = time_FairLP(idx) / time_Fair(idx);
end

% Reduced dimension count
r_count = (r_start:r_end)';

% Define color schemes for PCA, FairPCA, and FPCAviaSDR
color_pca_A = [1, 0, 0];  % Red for PCA (group A)
color_pca_B = [1, 0, 1];  % Magenta for PCA (group B)
color_fair_A = [0, 0, 1]; % Blue for FPCAviaEigOpt (group A)
color_fair_B = [0, 1, 1]; % Cyan for FPCAviaEigOpt (group B)
color_SDR = [0.9290 0.6940 0.1300];     % Yellow for FPCAviaSDR

% Plot the reconstruction error figure (PCA vs FPCAviaEigOpt)
figure
plot(r_count, error_M, "-s", "Color", color_pca_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_pca_A) % PCA (M)
hold on
plot(r_count, errorFair_M, "--d", "Color", color_fair_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_A) % FPCAviaEigOpt (M)
hold off
legend("PCA (M)", "FPCA (M)", "Location", "best", "FontSize", 30)
xlabel("r",'FontSize', 30)
ylabel("Error",'FontSize', 30)
grid on

% Overall error table
T1 = table(r_count, error_M, errorFair_M, ...
    errorFair_M ./ error_M,  ...
    'VariableNames', {'r', 'PCA_Error', 'FPCAviaEigOpt_Error', 'FPCAviaEigOpt/PCA'});
disp(T1);

% Loss figure (FPCA vs PCA)
figure
plot(r_count, rloss_A, "-s", "Color", color_pca_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_pca_A) % PCA (A)
hold on
plot(r_count, rloss_B, "-s", "Color", color_pca_B, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_pca_B) % PCA (B)
plot(r_count, rlossFair_A, "-d", "Color", color_fair_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_A) % FPCAviaEigOpt (A)
plot(r_count, rlossFair_B, "--d", "Color", color_fair_B, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_B) % FPCAviaEigOpt (B)
hold off
legend("PCA (A)", "PCA (B)", "FPCA (A)", "FPCA (B)", "Location", "best", "FontSize", 30)
xlabel("r",'FontSize', 30)
ylabel("Loss",'FontSize', 30)
grid on

% Runtime table
T1 = table(r_count, time_pca, time_Fair, time_FairLP, time_ratio1, time_ratio2,...
    'VariableNames', {'r', 'PCA', 'FPCAviaEigOpt', 'FPCAviaSDR', 'Time_Ratio1','Time_Ratio2'});
disp(T1)

% LossError table
T2 = table(r_count, abs(1-rloss_A./rloss_B), abs(1 - rlossFair_A ./ rlossFair_B), abs(1 - rloss_A_SDR ./ rloss_B_SDR), ...
    'VariableNames', {'r', 'Rloss_err_PCA', 'LossRatio_err_EigOpt', 'LossRatio_err_SDR'});
disp(T2)

