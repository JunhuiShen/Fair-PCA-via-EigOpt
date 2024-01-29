clc; close;clear; rng("default"); warning('off','all');warning;

[X, XA, XB] = creditProcess();
 % [X, XA, XB] = LFWProcess();
% [X, XA, XB] = bankProcess();
% [X,XA,XB] = cropProcess();

r_total = 10; 

d = size(XA,2);
na = size(XA,1);
nb = size(XB,1);

% Vanilla loss
loss_XA = zeros(r_total,1);
loss_XB = zeros(r_total,1);
loss_XAoverXBpca = zeros(r_total,1);

% Fair loss using trace definition
lossFair_XA = zeros(r_total,1);
lossFair_XB = zeros(r_total,1);
lossFair_XAoverXB = zeros(r_total,1);

% parameters of the mw algorithm
eta = 1;
T = 20; 

% Fair loss of Samadi's algorithm
z_last = zeros(r_total, 1);
z = zeros(r_total, 1);

loss_XAsamadi = zeros(r_total,1);
loss_XBsamadi = zeros(r_total,1);
loss_XAoverXBsamadi = zeros(r_total,1);

time_pca = zeros(r_total,1);
time_fair = zeros(r_total,1);
time_samadi = zeros(r_total,1);

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

    % Fair PCA
    tic
    [V,Ha,Hb] = fpca(XA, XB, ell, 10^(-8));
    time_fair(ell) = toc;

    % Fair PCA's average loss on A and B using Samadi's definition
    approx_XA = XA * (V * V');
    approx_XB = XB * (V * V');
    lossFair_XA(ell) = loss(XA,approx_XA,ell);
    lossFair_XB(ell) = loss(XB,approx_XB,ell);
    lossFair_XAoverXB(ell) = lossFair_XA(ell)/lossFair_XB(ell);

    % Fair PCA: Samadi
    tic
    Psamadi = fpca_samadi(XA,XB,ell,eta,T);
    time_samadi(ell) = toc;
    
    approxFair_XAsamadi = XA * Psamadi;
    approxFair_XBsamadi = XB * Psamadi;

    % Samadi's Fair PCA's average loss on A and B
    loss_XAsamadi(ell) = loss(XA,approxFair_XAsamadi,ell);
    loss_XBsamadi(ell) = loss(XB,approxFair_XBsamadi,ell);
    loss_XAoverXBsamadi(ell) = loss_XAsamadi(ell)/loss_XBsamadi(ell); 
end

r_count = 1:r_total;
r_count = r_count';

T = table(r_count,lossFair_XA,lossFair_XB,lossFair_XAoverXB);
T = table(r_count,loss_XAsamadi,loss_XBsamadi,loss_XAoverXBsamadi);
T = table(r_count,loss_XAoverXBpca,lossFair_XAoverXB,loss_XAoverXBsamadi)
T = table(r_count,time_pca,time_fair,time_samadi)

% plot the fairness figure
figure
x = r_count;
y1 = loss_XAoverXBpca;
y2 = lossFair_XAoverXB;
y3 = loss_XAoverXBsamadi;
% plot(x,y1,'--gs',...
%     'LineWidth',2,...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','b',...
%     'MarkerFaceColor',[0.5,0.5,0.5])
% hold on
plot(x,y2,'-rs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold on
plot(x,y3,'-go',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','c',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold off
legend("loss ratio of our FairPCA","loss ratio of Samadi's FairPCA")
xlabel("Number of dimensions")
ylabel("Loss")
title("Loss of algorithms")

% plot the time figure
figure
x = r_count; 
y1 = time_pca; plot(x,y1,"b--|",'LineWidth',3);
hold on
y2 = time_fair; plot(x,y2,"r-s",'LineWidth',3);
hold on
y3 = time_samadi; plot(x,y3,"g-o",'LineWidth',3);
hold off
legend("time PCA", "time Fair","time Samadi")
xlabel("Number of dimensions")
ylabel("Running time")
title("Running time of algorithms")
