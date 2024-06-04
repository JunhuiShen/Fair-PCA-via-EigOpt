clc; close all; clear; rng("default"); warning('off','all'); format long

% Parameters
na = 35; 
nb = 35;
A_mean = [0, 0]; 
B_mean = [0, 0]; 
cov_matrix1 = [1, 0.8; 0.8, 1]; % Covariance matrix for A
cov_matrix2 = [0.7,-0.6;-0.6,0.7]; % Covariance matrix for B
r = 1; % Reduce to 1 dimension
tol = 10^(-8);

% Generate synthetic data
A = mvnrnd(A_mean, cov_matrix1, na);
B = mvnrnd(B_mean, cov_matrix2, nb);
M = [A; B];

% Vanilla PCA
coeff = pca(M, 'NumComponents', r);
approx_Apca = A * (coeff * coeff');
approx_Bpca = B * (coeff * coeff');
approx_Mpca = M * (coeff * coeff');

% Fair PCA
U = fpca_Eigenvalue_Optimization(A, B, r, tol);
approx_A = A * (U * U');
approx_B = B * (U * U');
approx_M = M * (U * U');

% Determine line endpoints for plotting
approx_Mpca1 = approx_Mpca(:,1);
approx_Mpca2 = approx_Mpca(:,2);
[leftmost_approx_Mpca, idx1] = min(approx_Mpca1);
[rightmost_approx_Mpca, idx2] = max(approx_Mpca1);
xline1 = [leftmost_approx_Mpca, rightmost_approx_Mpca];
yline1 = [approx_Mpca2(idx1), approx_Mpca2(idx2)];

approx_M1 = approx_M(:,1);
approx_M2 = approx_M(:,2);
[leftmost_approx_M, idx1] = min(approx_M1);
[rightmost_approx_M, idx2] = max(approx_M1);
xline2 = [leftmost_approx_M, rightmost_approx_M];
yline2 = [approx_M2(idx1), approx_M2(idx2)];

% Plotting
figure
scatter(A(:,1), A(:,2), 100, 'g', 'o', 'filled');
hold on 
scatter(B(:,1), B(:,2), 100, 'c', 's', 'filled');
plot(approx_Mpca(:,1), approx_Mpca(:,2), 'LineWidth', 2, 'Color', [0.4940 0.1840 0.5560]);
plot(approx_M(:,1), approx_M(:,2), 'LineWidth', 2, 'Color', 'r');
hold off
xlabel('x')
ylabel('y')
legend('A', 'B', 'Vanilla PCA', 'Fair PCA')
title('Comparison between Vanilla PCA and Fair PCA')

% Save the figure as an EPS file
% print('-depsc2', 'synthetic_data.eps');

% Evaluation
loss_A = 1 / na * norm(A - approx_Apca, 'fro');
loss_B = 1 / nb * norm(B - approx_Bpca, 'fro');

% Vanilla PCA reconstruction loss
rloss_A = rloss(A, approx_Apca, r);
rloss_B = rloss(B, approx_Bpca, r);
rloss_AoverBpca = rloss_A / rloss_B;

% Reconstruction loss of Fair PCA via eigenvalue optimization
rlossFair_A = rloss(A, approx_A, r);
rlossFair_B = rloss(B, approx_B, r);
rlossFair_AoverB = rlossFair_A / rlossFair_B;

% Create the table
results = table(rloss_A, rloss_B, rloss_AoverBpca, rlossFair_A, rlossFair_B, rlossFair_AoverB, ...
    'VariableNames', {'rloss_A', 'rloss_B', 'rloss_AoverBpca', 'rlossFair_A', 'rlossFair_B', 'rlossFair_AoverB'});

% Display the table
disp(results);