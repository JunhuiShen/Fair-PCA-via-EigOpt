clc; close;clear; rng("default"); warning('off','all');warning;

% parameter
N = 35; 
XA_mean = [0, 0];
XB_mean = [0, 0];
cov_matrix1 = [1, -0.8; -0.8, 1];
cov_matrix2 = [1, 0.8; 0.8, 1];
r = 1; tol=10^(-8);

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

figure
scatter(XA(:,1),XA(:,2),100,"g","o","filled");
hold on 
scatter(XB(:,1),XB(:,2),100,"c","s","filled");
hold on
scatter(approx_Xpca(:,1),approx_Xpca(:,2),100,[0.4940 0.1840 0.5560],"*");
hold on
scatter(approx_X(:,1),approx_X(:,2),100,"r","o");
hold off
xlabel("x")
ylabel("y")
legend("X_A","X_B","Vanilla PCA","Fair PCA")
title("Comparison between Vanilla PCA and Fair PCA")
% print -depsc synthetic_data
