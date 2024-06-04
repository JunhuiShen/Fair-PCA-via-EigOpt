clc; close all; clear; rng("default"); warning('off','all');

% Data set
[M, A, B] = bank_marketing(); % Load the Bank Marketing dataset
% [M, A, B] = default_credit(); % Load the Default of Credit Card dataset
% [M, A, B] = crop_mapping(); % Load the Crop Mapping dataset
% [M, A, B] = LFW(); % Load the LFW dataset

% Total reduced dimension
r_total = 10; 

% Extract the dimension
d = size(A, 2);
na = size(A, 1);
nb = size(B, 1);

% Initialize reconstruction loss arrays
rloss_A = zeros(r_total, 1);
rloss_B = zeros(r_total, 1);
rlossFair_A = zeros(r_total, 1);
rlossFair_B = zeros(r_total, 1);
rloss_A_LP = zeros(r_total, 1);
rloss_B_LP = zeros(r_total, 1);

% Parameters
tol = 1e-8;
eta = 1;
T = 20;

% Initialize runtime arrays
time_pca = zeros(r_total, 1);
time_Fair = zeros(r_total, 1);
time_FairLP = zeros(r_total, 1);
time_ratio = zeros(r_total, 1);

for r = 1:r_total
    % PCA 
    tic;
    coeff = pca(M, "NumComponents", r); % Perform PCA
    time_pca(r) = toc;
    
    % Projection of PCA
    approx_Apca = A * (coeff * coeff');
    approx_Bpca = B * (coeff * coeff');
    
    % Reconstruction loss for PCA
    rloss_A(r) = rloss(A, approx_Apca, r); % rloss is already divided by the sample size
    rloss_B(r) = rloss(B, approx_Bpca, r);

    % FairPCAviaEigOpt
    tic;
    U = FairPCAviaEigOpt(A, B, r, tol); % Perform FairPCAviaEigOpt
    time_Fair(r) = toc;

    % Projection of FairPCAviaEigOpt
    approx_A = A * (U * U');
    approx_B = B * (U * U');

    % Reconstruction loss for FairPCAviaEigOpt
    rlossFair_A(r) = rloss(A, approx_A, r);
    rlossFair_B(r) = rloss(B, approx_B, r);

    % FairPCAviaLP
    tic;
    P_LP = FairPCAviaLP(A, B, r, eta, T); % Perform FairPCAviaLP
    time_FairLP(r) = toc;
    
    % Projection of FairPCAviaLP
    approxFair_A_LP = A * P_LP;
    approxFair_B_LP = B * P_LP;

    % Reconstruction loss for FairPCAviaLP
    rloss_A_LP(r) = rloss(A, approxFair_A_LP, r); 
    rloss_B_LP(r) = rloss(B, approxFair_B_LP, r); 

    % Runtime ratio between FairPCAviaEigOpt and FairPCAviaLP
    time_ratio(r) = time_FairLP(r) / time_Fair(r);
end

% Reduced dimension count
r_count = (1:r_total)';

% Plot the rloss ratio figure
figure
plot(r_count, rlossFair_A ./ rlossFair_B, '-rs', 'LineWidth', 2, ...
    'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.5, 0.5, 0.5])
hold on
plot(r_count, rloss_A_LP ./ rloss_B_LP, '-go', 'LineWidth', 2, ...
    'MarkerSize', 10, 'MarkerEdgeColor', 'c', 'MarkerFaceColor', [0.5, 0.5, 0.5])
hold off
legend("FairPCAviaEigOpt", "FairPCAviaLP", 'Location', 'best')
xlabel("Number of Reduced Dimensions")
ylabel("Average Reconstruction Loss Ratio")
title("Average Reconstruction Loss Ratio: FairPCAviaEigOpt vs FairPCAviaLP")
grid on
% print('-depsc', 'ratio')

% Plot the rloss figure
figure
plot(r_count, rloss_A, '-rs', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r')
hold on
plot(r_count, rloss_B, '-go', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerEdgeColor', 'c', 'MarkerFaceColor', 'g')
plot(r_count, rlossFair_A, '-ms', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'm')
plot(r_count, rlossFair_B, '--bo', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b')
hold off
legend("PCA (A)", "PCA (B)", "FairPCAviaEigOpt (A)", "FairPCAviaEigOpt (B)", 'Location', 'best')
xlabel("Number of Reduced Dimensions")
ylabel("Average Reconstruction Loss")
title("Average Reconstruction Loss: FairPCAviaEigOpt vs PCA")
grid on
% print('-depsc', 'rloss')

% Display runtime table
T = table(r_count, time_pca, time_Fair, time_FairLP, time_ratio);

% Plot the efficiency figure
figure
b = bar(r_count, [time_pca, time_Fair, time_FairLP], 'grouped');
hold on
% Adjust the y-axis limit to ensure annotations fit
ylim([0, max(time_FairLP) * 1.1])
% Add text annotations to display the speedup ratios
for i = 1:r_total
    text(b(3).XData(i), b(3).YData(i) + 0.01, sprintf('%.1fx', time_ratio(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'Color', 'black', 'FontWeight', 'bold')
end
hold off
legend("PCA", "FairPCAviaEigOpt", "FairPCAviaLP", 'Location', 'best')
xlabel("Number of Reduced Dimensions")
ylabel("Runtime (seconds)")
title("Runtime:  FairPCAviaEigOpt vs FairPCAviaLP vs PCA")
grid on
% print('-depsc', 'runtime')