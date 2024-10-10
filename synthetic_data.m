clc; close all; clear; rng("default"); warning('off','all');

% Parameters
na = 50;
nb = 20;
A_mean = [0, 0]; 
B_mean = [0, 0]; 
cov_matrix1 = [1, 0.5; 0.5, 1.2]; % Covariance matrix for A 
cov_matrix2 = [1.2, -0.5; -0.5, 1]; % Covariance matrix for B 
r = 1; 
tol = 10^(-8);

% Generate synthetic data
A = mvnrnd(A_mean, cov_matrix1, na);
B = mvnrnd(B_mean, cov_matrix2, nb);
M = [A; B];

% PCA
coeff = pca(M, "NumComponents", r);
proj_Apca = A * (coeff * coeff');
proj_Bpca = B * (coeff * coeff');

% Fair PCA
U = FPCAviaEigOpt(A, B, r, tol);
proj_AFair = A * (U * U');
proj_BFair = B * (U * U');

% Set a fixed scaling factor for line length
line_scale = 3;

figure;

% Scatter plot of original dataset and PCA vector
subplot(2, 3, 1);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled'); % Scatter plot for group A
hold on
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled'); % Scatter plot for group B
plot([-line_scale line_scale] * coeff(1,1), [-line_scale line_scale] * coeff(2,1), 'k', 'LineWidth', 1.5); % PCA vector
xlabel("Feature 1", 'FontSize', 12); ylabel("Feature 2", 'FontSize', 12);
legend("A", "B", "PCA");
title("Dataset and PCA","FontSize",12);
grid on; axis equal;
% Get axis limits for PCA plots
x_limits_PCA = xlim;
y_limits_PCA = ylim;

% PCA Projection of Group A
subplot(2, 3, 2);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled');
hold on
scatter(proj_Apca(:,1), proj_Apca(:,2), 100, 'k', '*'); % Projection points for Group A
xlabel("Feature 1", 'FontSize', 12); ylabel("Feature 2", 'FontSize', 12);
legend("A", "Projected A");
title("PCA Projection of A","FontSize",12);
grid on; axis equal;
% Use the same axis limits as subplot 1
xlim(x_limits_PCA); ylim(y_limits_PCA);

% PCA Projection of Group B 
subplot(2, 3, 3);
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled');
hold on
scatter(proj_Bpca(:,1), proj_Bpca(:,2), 100, 'k', '*'); % Projection points for Group B
xlabel("Feature 1", 'FontSize', 12); ylabel("Feature 2", 'FontSize', 12);
legend("B", "Projected B");
title("PCA Projection of B","FontSize",12);
grid on; axis equal;
% Use the same axis limits as subplot 1
xlim(x_limits_PCA); ylim(y_limits_PCA);
% print("synthetic_data",'-depsc2');

% figure;
% Dataset and Fair PCA vector 
subplot(2, 3, 4);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled');
hold on
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled');
plot([-line_scale line_scale] * U(1,1), [-line_scale line_scale] * U(2,1), 'k', 'LineWidth', 1.5); % Fair PCA vector
xlabel("Feature 1", 'FontSize', 12); ylabel("Feature 2", 'FontSize', 12);
legend("A", "B", "FPCA");
title("Dataset and FPCA","FontSize",12);
grid on; axis equal;
% Get axis limits for FPCA plots
x_limits_FPCA = xlim;
y_limits_FPCA = ylim;

% Fair PCA Projection of Group A 
subplot(2, 3, 5);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled');
hold on
scatter(proj_AFair(:,1), proj_AFair(:,2), 100, 'k', '*'); % Projection points for Group A
xlabel("Feature 1", 'FontSize', 12); ylabel("Feature 2", 'FontSize', 12);
legend("A", "Projected A");
title("FPCA Projection of A","FontSize",12);
grid on; axis equal;
% Use the same axis limits as subplot 4
xlim(x_limits_FPCA); ylim(y_limits_FPCA);

% Fair PCA Projection of Group B 
subplot(2, 3, 6);
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled');
hold on
scatter(proj_BFair(:,1), proj_BFair(:,2), 100, 'k', '*'); % Projection points for Group B
xlabel("Feature 1", 'FontSize', 12); ylabel("Feature 2", 'FontSize', 12);
legend("B", "Projected B");
title("FPCA Projection of B","FontSize",12);
grid on; axis equal;
% Use the same axis limits as subplot 4
xlim(x_limits_FPCA); ylim(y_limits_FPCA);
% print("synthetic_data2",'-depsc2');
