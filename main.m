clc; close all; clear; rng("default"); warning("off","all");

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
d = size(A, 2);
na = size(A, 1);
nb = size(B, 1);
n = na + nb;

% Initialize reconstruction error arrays
error_A = zeros(r_total, 1);
error_B = zeros(r_total, 1);
error_M = zeros(r_total, 1);
errorFair_A = zeros(r_total, 1);
errorFair_B = zeros(r_total, 1);
errorFair_M = zeros(r_total, 1);
error_A_LP = zeros(r_total, 1);
error_B_LP = zeros(r_total, 1);
error_M_LP = zeros(r_total, 1);

% Initialize reconstruction loss arrays
rloss_A = zeros(r_total, 1);
rloss_B = zeros(r_total, 1);
rlossFair_A = zeros(r_total, 1);
rlossFair_B = zeros(r_total, 1);
rloss_A_LP = zeros(r_total, 1);
rloss_B_LP = zeros(r_total, 1);

% Parameters
tol = 1e-8; % Tolerance for optimization
eta = 1; % Parameter for Fair PCA via LP
T = 20; % Number of iterations for Fair PCA via LP

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
    approx_Apca = A * (coeff * coeff');
    approx_Bpca = B * (coeff * coeff');

    % Reconstruction error for PCA
    error_A(idx) = error1(A,approx_Apca);
    error_B(idx) = error1(B,approx_Bpca);
    error_M(idx) = error1(M,M * (coeff * coeff'));

    % Reconstruction loss for PCA
    rloss_A(idx) = rloss(A, approx_Apca, r); % rloss is already divided by the sample size
    rloss_B(idx) = rloss(B, approx_Bpca, r);

    % FairPCAviaEigOpt
    tic;
    [U] = FairPCAviaEigOpt(A, B, r, tol); % Perform FairPCAviaEigOpt
    time_Fair(idx) = toc;

    % Projection of FairPCAviaEigOpt
    approx_A = A * (U * U');
    approx_B = B * (U * U');

    % Reconstruction error for FairPCAviaEigOpt
    errorFair_A(idx) = error1(A,approx_A);
    errorFair_B(idx) = error1(B,approx_B);
    errorFair_M(idx) = error1(M,M * (U * U'));

    % Reconstruction loss for FairPCAviaEigOpt
    rlossFair_A(idx) = rloss(A, approx_A, r);
    rlossFair_B(idx) = rloss(B, approx_B, r);

    % FairPCAviaLP
    tic;
    P_LP = FairPCAviaLP(A, B, r, eta, T); % Perform FairPCAviaLP
    time_FairLP(idx) = toc;

    % Projection of FairPCAviaLP
    approxFair_A_LP = A * P_LP;
    approxFair_B_LP = B * P_LP;

    % Reconstruction error for FairPCAviaLP
    error_A_LP(idx) = error1(A, approxFair_A_LP); 
    error_B_LP(idx) = error1(B, approxFair_B_LP); 
    error_M_LP(idx) = error1(M, M * P_LP); 

    % Reconstruction loss for FairPCAviaLP
    rloss_A_LP(idx) = rloss(A, approxFair_A_LP, r); 
    rloss_B_LP(idx) = rloss(B, approxFair_B_LP, r); 

    % Runtime ratio between FairPCAviaEigOpt and FairPCAviaLP
    time_ratio1(idx) = time_Fair(idx) / time_pca(idx);
    time_ratio2(idx) = time_FairLP(idx) / time_Fair(idx);

    % % Save plot if iteration matches save_iter
    % if r == 7
    %     print('-depsc2', sprintf('sample_phit.eps', r)); % Save EPS
    % end
end

% Reduced dimension count
r_count = (r_start:r_end)';

% Define color schemes for PCA, FairPCA, and FairPCAviaLP
color_pca_A = [1, 0, 0];  % Red for PCA (group A)
color_pca_B = [1, 0, 1];  % Magenta for PCA (group B)
color_fair_A = [0, 0, 1]; % Blue for FairPCAviaEigOpt (group A)
color_fair_B = [0, 1, 1]; % Cyan for FairPCAviaEigOpt (group B)
color_lp = [0.9290 0.6940 0.1250];     % Yellow for FairPCAviaLP

% Plot the reconstruction error figure (PCA vs FairPCAviaEigOpt)
figure
plot(r_count, error_M, "-s", "Color", color_pca_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_pca_A) % PCA (M)
hold on
plot(r_count, errorFair_M, "--d", "Color", color_fair_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_A) % FairPCAviaEigOpt (M)
hold off
legend("PCA (All)", "FairPCAviaEigOpt (All)", "Location", "best")
xlabel("Number of Reduced Dimensions")
ylabel("Reconstruction Error")
title("Reconstruction Error: PCA vs FairPCAviaEigOpt")
grid on
% print("-depsc", "error")

% Plot the average reconstruction loss figure (FairPCAviaEigOpt vs PCA)
figure
plot(r_count, rloss_A, "-s", "Color", color_pca_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_pca_A) % PCA (A)
hold on
plot(r_count, rloss_B, "-s", "Color", color_pca_B, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_pca_B) % PCA (B)
plot(r_count, rlossFair_A, "-d", "Color", color_fair_A, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_A) % FairPCAviaEigOpt (A)
plot(r_count, rlossFair_B, "--d", "Color", color_fair_B, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_B) % FairPCAviaEigOpt (B)
hold off
legend("PCA (Female)", "PCA (Male)", "FairPCAviaEigOpt (Female)", "FairPCAviaEigOpt (Male)", "Location", "best")
xlabel("Number of Reduced Dimensions")
ylabel("Average Reconstruction Loss")
title("Average Reconstruction Loss: FairPCAviaEigOpt vs PCA")
grid on
% print("-depsc", "rloss")

% Plot the rloss ratio figure (FairPCAviaEigOpt vs FairPCAviaLP)
figure
plot(r_count, rlossFair_A ./ rlossFair_B, "-d", "Color", color_fair_A, "LineWidth", 2, ...  
    "MarkerSize", 10, "MarkerEdgeColor", "k", "MarkerFaceColor", color_fair_A) % Blue for FairPCAviaEigOpt ratio
hold on
plot(r_count, rloss_A_LP ./ rloss_B_LP, "-o", "Color", color_lp, "LineWidth", 2, "MarkerSize", 10, ...
    "MarkerEdgeColor", "k", "MarkerFaceColor", color_lp) % Yellow for FairPCAviaLP ratio
hold off
legend("FairPCAviaEigOpt", "FairPCAviaLP", "Location", "best")
xlabel("Number of Reduced Dimensions")
ylabel("Average Reconstruction Loss Ratio")
title("Average Reconstruction Loss Ratio: FairPCAviaEigOpt vs FairPCAviaLP")
grid on
% print("-depsc", "ratio")


T = table(r_count, time_pca, time_Fair, time_FairLP, time_ratio1, time_ratio2,...
    'VariableNames', {'r', 'PCA', 'FairPCAviaEigOpt', 'FairPCAviaLP', 'Time_Ratio1','Time_Ratio2'});
disp(T)

% % Plot the runtime figure (FairPCAviaEigOpt vs FairPCAviaLP vs PCA)
% figure
% b = bar(r_count, [time_pca, time_Fair, time_FairLP], "grouped");
% b(1).FaceColor = color_pca_A;  % Red for PCA runtime
% b(2).FaceColor = color_fair_A; % Blue for FairPCAviaEigOpt runtime
% b(3).FaceColor = color_lp;     % Yellow for FairPCAviaLP runtime
% 
% hold on
% ylim([0, max(time_FairLP) * 1.1])
% for i = 1:r_total
%     text(b(3).XData(i), b(3).YData(i) + 0.01, sprintf("%.1fx", time_ratio2(i)), ...
%         "HorizontalAlignment", "center", "VerticalAlignment", "bottom", "FontSize", 10, "Color", "black", "FontWeight", "bold")
% end
% hold off
% legend("PCA", "FairPCAviaEigOpt", "FairPCAviaLP", "Location", "best")
% xlabel("Number of Reduced Dimensions")
% ylabel("Runtime (seconds)")
% title("Runtime:  FairPCAviaEigOpt vs FairPCAviaLP vs PCA")
% grid on
% % print("-depsc", "runtime")

