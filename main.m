clc; close;clear; rng("default"); warning('off','all');warning;

% [X, XA, XB] = creditProcess();
 % [X, XA, XB] = LFWProcess();
% [X, XA, XB] = bankProcess();
[X,XA,XB] = cropProcess();

r_total = 10; 

d = size(XA,2);
na = size(XA,1);
nb = size(XB,1);

% Vanilla loss
loss_XA = zeros(r_total,1);
loss_XB = zeros(r_total,1);
loss_XAoverXBpca = zeros(r_total,1);

% Fair loss of Fair PCA via convex optimization
lossFair_XA = zeros(r_total,1);
lossFair_XB = zeros(r_total,1);
lossFair_max = zeros(r_total,1);
lossFair_XAoverXB = zeros(r_total,1);

% parameters of the mw algorithm
eta = 1;
T = 20; 
z_last = zeros(r_total, 1);
z = zeros(r_total, 1);

% Fair loss of Fair PCA via convex optimization
loss_XA_LP = zeros(r_total,1);
loss_XB_LP = zeros(r_total,1);
loss_LP_max = zeros(r_total,1);
loss_XAoverXB_LP = zeros(r_total,1);

% Time
time_pca = zeros(r_total,1);
time_FairConvex = zeros(r_total,1);
time_FairLP = zeros(r_total,1);

for ell=1:r_total

    % Vanilla PCA 
    tic
    coeff = pca(X,"NumComponents",ell);
    time_pca(ell) = toc;
    
    % Vanilla PCA's average loss on A and B
    approx_XApca = XA * (coeff * coeff');
    approx_XBpca = XB * (coeff * coeff');
    loss_XA(ell) = loss(XA,approx_XApca,ell);
    loss_XB(ell) = loss(XB,approx_XBpca,ell);
    loss_XAoverXBpca(ell) = loss_XA(ell)/loss_XB(ell);

    % Fair PCA via convex optimization
    tic
    U = fpca(XA, XB, ell, 10^(-8));
    time_FairConvex(ell) = toc;

    % the average loss on A and B of Fair PCA via convex optimization
    approx_XA = XA * (U * U');
    approx_XB = XB * (U * U');
    lossFair_XA(ell) = loss(XA,approx_XA,ell);
    lossFair_XB(ell) = loss(XB,approx_XB,ell);

    lossFair_max(ell) = max(lossFair_XA(ell),lossFair_XB(ell));
    lossFair_XAoverXB(ell) = lossFair_XA(ell)/lossFair_XB(ell);

    % Fair PCA via LP
    tic
    P_LP = fpca_LP(XA,XB,ell,eta,T);
    time_FairLP(ell) = toc;

    approxFair_XA_LP = XA * P_LP;
    approxFair_XB_LP = XB * P_LP;

    % the average loss on A and B of Fair PCA via LP
    loss_XA_LP(ell) = loss(XA,approxFair_XA_LP,ell);
    loss_XB_LP(ell) = loss(XB,approxFair_XB_LP,ell);

    loss_LP_max(ell) = max(loss_XA_LP(ell),loss_XB_LP(ell));
    loss_XAoverXB_LP(ell) = loss_XA_LP(ell)/loss_XB_LP(ell); 
end

r_count = 1:r_total;
r_count = r_count';

T = table(r_count,lossFair_XA,lossFair_XB,lossFair_XAoverXB);
T = table(r_count,loss_XA_LP,loss_XB_LP,loss_XAoverXB_LP);
T = table(r_count,lossFair_max,loss_LP_max);
T = table(r_count,loss_XAoverXBpca,lossFair_XAoverXB,loss_XAoverXB_LP);
T = table(r_count,time_pca,time_FairConvex,time_FairLP);

% plot the loss ratio figure
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
legend("FairPCA via convex optimization","FairPCA via LP")
xlabel("Number of dimensions")
ylabel("Loss ratio")
title("Loss Ratio")
% print -depsc newfigure1

% plot the fairness figure
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
legend("FairPCA via convex optimization","FairPCA via LP")
xlabel("Number of dimensions")
ylabel("Fairness measure")
title("Fairness measure")
% print -depsc newfigure2

% plot the time figure
figure
x = r_count; 
y1 = time_pca; plot(x,y1,"b--|",'LineWidth',3);
hold on
y2 = time_FairConvex; plot(x,y2,"r-s",'LineWidth',3);
hold on
y3 = time_FairLP; plot(x,y3,"g-o",'LineWidth',3);
hold off
legend("Vanilla PCA", "FairPCA via convex optimization","FairPCA via LP")
xlabel("Number of dimensions")
ylabel("Running time")
title("Running time of algorithms")
% print -depsc newfigure3
