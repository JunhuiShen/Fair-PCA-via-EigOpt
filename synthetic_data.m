clc; close all; clear; rng("default"); warning('off','all');

% Parameters
na = 50; % Larger sample size for A
nb = 20; % Smaller sample size for B
A_mean = [0, 0]; 
B_mean = [0, 0]; 
cov_matrix1 = [1, 0.5; 0.5, 1.2]; % Covariance matrix for A 
cov_matrix2 = [1.2, -0.5; -0.5, 1]; % Covariance matrix for B 
r = 1; % Reduce to 1 dimension
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
U = FairPCAviaEigOpt(A, B, r, tol);
proj_AFair = A * (U * U');
proj_BFair = B * (U * U');

% Set a fixed scaling factor for line length
line_scale = 3;

% First figure for (a), (b), (c)
figure;

% (a) Scatter plot of original dataset and PCA vector (unchanged)
subplot(2, 3, 1);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled'); % Scatter plot for group A
hold on
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled'); % Scatter plot for group B
plot([-line_scale line_scale] * coeff(1,1), [-line_scale line_scale] * coeff(2,1), 'k', 'LineWidth', 1.5); % PCA vector
xlabel("First attribute"); ylabel("Second attribute");
legend("A", "B", "PCA");
title("(a) Dataset and PCA");
grid on; axis equal;

% (b) PCA Projection of Group A (points)
subplot(2, 3, 2);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled');
hold on
scatter(proj_Apca(:,1), proj_Apca(:,2), 100, 'k', '*'); % Projection points for Group A
xlabel("First attribute"); ylabel("Second attribute");
legend("A", "Projected A");
title("(b) PCA projection of A");
grid on; axis equal;

% (c) PCA Projection of Group B (points)
subplot(2, 3, 3);
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled');
hold on
scatter(proj_Bpca(:,1), proj_Bpca(:,2), 100, 'k', '*'); % Projection points for Group B
xlabel("First attribute"); ylabel("Second attribute");
legend("B", "Projected B");
title("(c) PCA projection of B");
grid on; axis equal;

% Save the first figure as EPS file for (a), (b), (c)
% print('-depsc2', 'synthetic_data.eps');

% Second figure for (d), (e), (f)
figure;

% (d) Scatter plot of original dataset and Fair PCA vector 
subplot(2, 3, 4);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled');
hold on
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled');
plot([-line_scale line_scale] * U(1,1), [-line_scale line_scale] * U(2,1), 'k', 'LineWidth', 1.5); % Fair PCA vector
xlabel("First attribute"); ylabel("Second attribute");
legend("A", "B", "Fair PCA");
title("(d) Dataset and Fair PCA");
grid on; axis equal;

% (e) Fair PCA Projection of Group A (points)
subplot(2, 3, 5);
scatter(A(:,1), A(:,2), 100, [0.9290 0.6940 0.1250], 'filled');
hold on
scatter(proj_AFair(:,1), proj_AFair(:,2), 100, 'k', '*'); % Projection points for Group A
xlabel("First attribute"); ylabel("Second attribute");
legend("A", "Projected A");
title("(e) Fair PCA projection of A");
grid on; axis equal;

% (f) Fair PCA Projection of Group B (points)
subplot(2, 3, 6);
scatter(B(:,1), B(:,2), 100, [0 0.4470 0.7410], 'filled');
hold on
scatter(proj_BFair(:,1), proj_BFair(:,2), 100, 'k', '*'); % Projection points for Group B
xlabel("First attribute"); ylabel("Second attribute");
legend("B", "Projected B");
title("(f) Fair PCA projection of B");
grid on; axis equal;

% Save the second figure as EPS file for (d), (e), (f)
% print('-depsc2', 'synthetic_data2.eps');


