clc; close all; clear; rng("default"); warning('off','all');

% Parameters
na = 50; % Larger sample size for A
nb = 20; % Smaller sample size for B
A_mean = [0, 0]; 
B_mean = [0, 0]; 
cov_matrix1 = [0.5, 0.5; 0.5, 0.5]; % Covariance matrix for A 
cov_matrix2 = [1, -0.3; -0.3, 0.5];    % Covariance matrix for B 
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
% print('-depsc2', 'synthetic_data2.eps');

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