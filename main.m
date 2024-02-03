clc; close;clear; rng("default"); warning('off','all');warning;

% Data set
[X, XA, XB] = bankProcess();
% [X, XA, XB] = creditProcess();
% [X, XA, XB] = LFWProcess();
% [X,XA,XB] = cropProcess();

% Total reduced dimension
r_total = 10; 

% Extract the dimension
d = size(XA,2);
na = size(XA,1);
nb = size(XB,1);

% Vanilla loss
loss_XA = zeros(r_total,1);
loss_XB = zeros(r_total,1);
loss_XAoverXBpca = zeros(r_total,1);

% Loss of Fair PCA via eigenvalue Optimization
lossFair_XA = zeros(r_total,1);
lossFair_XB = zeros(r_total,1);
lossFair_max = zeros(r_total,1);
lossFair_XAoverXB = zeros(r_total,1);

% Parameter of Fair PCA via eigenvalue Optimization
tol = 10^(-8);

% Loss of Fair PCA via LP
loss_XA_LP = zeros(r_total,1);
loss_XB_LP = zeros(r_total,1);
loss_LP_max = zeros(r_total,1);
loss_XAoverXB_LP = zeros(r_total,1);

% Parameters of Fair PCA via LP
eta = 1;
T = 20; 
z_last = zeros(r_total, 1);
z = zeros(r_total, 1);

% Efficiency
time_pca = zeros(r_total,1);
time_FairConvex = zeros(r_total,1);
time_FairLP = zeros(r_total,1);

for r=1:r_total

    % Vanilla PCA 
    tic
    coeff = pca(X,"NumComponents",r);
    time_pca(r) = toc;
    
    % Projection of Vanilla PCA
    approx_XApca = XA * (coeff * coeff');
    approx_XBpca = XB * (coeff * coeff');

    % The average loss on A and B of Vanilla PCA
    loss_XA(r) = loss(XA,approx_XApca,r);
    loss_XB(r) = loss(XB,approx_XBpca,r);
    loss_XAoverXBpca(r) = loss_XA(r)/loss_XB(r);

    % Fair PCA via eigenvalue Optimization
    tic
    U = fpca_Eigenvalue_Optimization(XA, XB, r,tol);
    time_FairConvex(r) = toc;

    % Projection of Fair PCA via eigenvalue Optimization
    approx_XA = XA * (U * U');
    approx_XB = XB * (U * U');

    % The average loss on A and B of Fair PCA via eigenvalue Optimization
    lossFair_XA(r) = loss(XA,approx_XA,r);
    lossFair_XB(r) = loss(XB,approx_XB,r);
    lossFair_max(r) = max(lossFair_XA(r),lossFair_XB(r));
    lossFair_XAoverXB(r) = lossFair_XA(r)/lossFair_XB(r);

    % Fair PCA via LP
    tic
    P_LP = fpca_LP(XA,XB,r,eta,T);
    time_FairLP(r) = toc;
    
    % Projection of Fair PCA via LP
    approxFair_XA_LP = XA * P_LP;
    approxFair_XB_LP = XB * P_LP;

    % The average loss on A and B of Fair PCA via LP
    loss_XA_LP(r) = loss(XA,approxFair_XA_LP,r);
    loss_XB_LP(r) = loss(XB,approxFair_XB_LP,r);
    loss_LP_max(r) = max(loss_XA_LP(r),loss_XB_LP(r));
    loss_XAoverXB_LP(r) = loss_XA_LP(r)/loss_XB_LP(r); 
end

% Make a list for each reduced dimension
r_count = 1:r_total;
r_count = r_count';

% Table of different comparison result
T = table(r_count,lossFair_XA,lossFair_XB,lossFair_XAoverXB);
T = table(r_count,loss_XA_LP,loss_XB_LP,loss_XAoverXB_LP);
T = table(r_count,lossFair_max,loss_LP_max);
T = table(r_count,loss_XAoverXBpca,lossFair_XAoverXB,loss_XAoverXB_LP);
T = table(r_count,time_pca,time_FairConvex,time_FairLP);

% Plot the loss ratio figure
figure
x = r_count;
y1 = lossFair_XAoverXB;
y2 = loss_XAoverXB_LP;
plot(x,y1,'-rs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold on
plot(x,y2,'-go',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','c',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold off
legend("FairPCA via eigenvalue optimization","FairPCA via LP")
xlabel("Number of reduced dimensions")
ylabel("Loss ratio")
title("Loss Ratio")
% print -depsc newfigure1

% Plot the fairness measure figure
figure
x = r_count;
y1 = lossFair_max;
y2 = loss_LP_max;
plot(x,y1,'-rs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold on
plot(x,y2,'-go',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','c',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold off
legend("FairPCA via eigenvalue optimization","FairPCA via LP")
xlabel("Number of reduced dimensions")
ylabel("Fairness measure")
title("Fairness measure")
% print -depsc newfigure2

% Plot the efficiency figure
figure
x = r_count; 
y1 = time_pca; plot(x,y1,"b--|",'LineWidth',3);
hold on
y2 = time_FairConvex; plot(x,y2,"r-s",'LineWidth',3);
hold on
y3 = time_FairLP; plot(x,y3,"g-o",'LineWidth',3);
hold off
legend("Vanilla PCA", "FPCA via eigenvalue optimization","FPCA via LP")
xlabel("Number of reduced dimensions")
ylabel("Running time")
title("Running time")
% print -depsc newfigure3
