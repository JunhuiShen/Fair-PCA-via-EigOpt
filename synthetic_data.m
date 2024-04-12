clc; close; clear;  rng("default");  warning('off','all');warning;

% parameter
N = 35; 
XA_mean = [0, 0, 0]; 
XB_mean = [0, 0, 0]; 
cov_matrix1 = [1, -0.8, 0.5; -0.8, 1, -0.5; 0.5, -0.5, 1]; 
cov_matrix2 = [1, 0.8, -0.2; 0.8, 1, 0.3; -0.2, 0.3, 1]; 
r = 1; tol = 10^(-8);

% Generate synthetic data
XA = mvnrnd(XA_mean, cov_matrix1, N);
XB = mvnrnd(XB_mean, cov_matrix2, N);
X = [XA;XB];

% Vanilla PCA
coeff = pca(X,"NumComponents",r);
approx_XApca = XA * (coeff * coeff');
approx_XBpca = XB * (coeff * coeff');
approx_Xpca = X * (coeff * coeff');

% Fair PCA
U = fpca_Eigenvalue_Optimization(XA, XB, r,tol);
approx_XA = XA *(U * U');
approx_XB = XB *(U * U');
approx_X = X *(U * U');

approx_Xpca1 = approx_Xpca(:,1);
approx_Xpca2 = approx_Xpca(:,2);
[leftmost_approx_Xpca,idx1] = min(approx_Xpca1);
[rightmost_approx_Xpca,idx2] = max(approx_Xpca1);
xline1 = [leftmost_approx_Xpca,rightmost_approx_Xpca];
yline1 = [approx_Xpca2(idx1),approx_Xpca2(idx2)];

approx_X1 = approx_X(:,1);
approx_X2 = approx_X(:,2);
[leftmost_approx_X,idx1] = min(approx_X1);
[rightmost_approx_X,idx2] = max(approx_X1);
xline2 = [leftmost_approx_X,rightmost_approx_X];
yline2 = [approx_X2(idx1),approx_X2(idx2)];

figure
scatter3(XA(:,1), XA(:,2), XA(:,3), 100, "g", "o", "filled");
hold on 
scatter3(XB(:,1), XB(:,2), XB(:,3), 100, "c", "s", "filled");
hold on
scatter3(approx_Xpca(:,1), approx_Xpca(:,2), approx_Xpca(:,3), 100, [0.4940 0.1840 0.5560], "*");
% plot3(approx_Xpca(:,1), approx_Xpca(:,2), approx_Xpca(:,3), "LineWidth", 2)
hold on
scatter3(approx_X(:,1), approx_X(:,2), approx_X(:,3), 100, "r", "o");
% plot3(approx_X(:,1), approx_X(:,2), approx_X(:,3), "LineWidth", 2)
hold off
xlabel("x")
ylabel("y")
zlabel("z")
legend("X_A","X_B","Vanilla PCA","Fair PCA")
title("Comparison between Vanilla PCA and Fair PCA")
