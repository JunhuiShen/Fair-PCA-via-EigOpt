clc; close; clear;  rng("default");  warning('off','all');warning;

% parameter
NA = 35; 
NB = 35;
A_mean = [0, 0, 0]; 
MB_mean = [0, 0, 0]; 
cov_matrix1 = [1, -0.8, 0.5; -0.8, 1, -0.5; 0.5, -0.5, 1]; 
cov_matrix2 = 0.5 * [1, 0.8, -0.2; 0.8, 1, 0.3; -0.2, 0.3, 1]; 
r = 1; tol = 10^(-8);

% Generate synthetic data
A = mvnrnd(A_mean, cov_matrix1, NA);
B = mvnrnd(MB_mean, cov_matrix2, NB);
M = [A;B];

% Vanilla PCA
coeff = pca(M,"NumComponents",r);
approx_Apca = A * (coeff * coeff');
approx_MBpca = B * (coeff * coeff');
approx_Mpca = M * (coeff * coeff');

% Fair PCA
U = fpca_Eigenvalue_Optimization(A, B, r,tol);
approx_A = A *(U * U');
approx_B = B *(U * U');
approx_M = M *(U * U');

approx_Mpca1 = approx_Mpca(:,1);
approx_Mpca2 = approx_Mpca(:,2);
[leftmost_approx_Mpca,idx1] = min(approx_Mpca1);
[rightmost_approx_Mpca,idx2] = max(approx_Mpca1);
xline1 = [leftmost_approx_Mpca,rightmost_approx_Mpca];
yline1 = [approx_Mpca2(idx1),approx_Mpca2(idx2)];

approx_M1 = approx_M(:,1);
approx_M2 = approx_M(:,2);
[leftmost_approx_M,idx1] = min(approx_M1);
[rightmost_approx_M,idx2] = max(approx_M1);
xline2 = [leftmost_approx_M,rightmost_approx_M];
yline2 = [approx_M2(idx1),approx_M2(idx2)];

figure
scatter3(A(:,1), A(:,2), A(:,3), 100, "g", "o", "filled");
hold on 
scatter3(B(:,1), B(:,2), B(:,3), 100, "c", "s", "filled");
hold on
% scatter3(approx_Mpca(:,1), approx_Mpca(:,2), approx_Mpca(:,3), 100, [0.4940 0.1840 0.5560], "*");
plot3(approx_Mpca(:,1), approx_Mpca(:,2), approx_Mpca(:,3), "LineWidth", 2)
hold on
% scatter3(approx_M(:,1), approx_M(:,2), approx_M(:,3), 100, "r", "o");
plot3(approx_M(:,1), approx_M(:,2), approx_M(:,3), "LineWidth", 2)
hold off
xlabel("x")
ylabel("y")
zlabel("z")
legend("A","B","Vanilla PCA","Fair PCA")
title("Comparison between Vanilla PCA and Fair PCA")
