clc; close;clear; rng("default"); warning('off','all');warning;format short

% Data set
[M, A, B] = bankProcess();
% [M, A, B] = creditProcess();
% [M,A,B] = cropProcess();
% [M, A, B] = LFWProcess();

% Total reduced dimension
r_total = 10; 

% Extract the dimension
d = size(A,2);
na = size(A,1);
nb = size(B,1);

% Vanilla PCA reconstruction rloss
rloss_A = zeros(r_total,1);
rloss_B = zeros(r_total,1);
rloss_AoverBpca = zeros(r_total,1);

% Reconstruction loss of Fair PCA via eigenvalue Optimization
rlossFair_A = zeros(r_total,1);
rlossFair_B = zeros(r_total,1);
rlossFair_max = zeros(r_total,1);
rlossFair_AoverB = zeros(r_total,1);

% Parameter of Fair PCA via eigenvalue Optimization
tol = 10^(-8);

% Reconstruction loss of Fair PCA via LP
rloss_A_LP = zeros(r_total,1);
rloss_B_LP = zeros(r_total,1);
rloss_LP_max = zeros(r_total,1);
rloss_AoverB_LP = zeros(r_total,1);

% Parameters of Fair PCA via LP
eta = 1;
T = 20; 
z_last = zeros(r_total, 1);
z = zeros(r_total, 1);

% Runtime
time_pca = zeros(r_total,1);
time_Fair = zeros(r_total,1);
time_FairLP = zeros(r_total,1);
time_ratio = zeros(r_total,1);

for r=1:r_total

    % Vanilla PCA 
    tic
    coeff = pca(M,"NumComponents",r);
    time_pca(r) = toc;
    
    % Projection of Vanilla PCA
    approx_Apca = A * (coeff * coeff');
    approx_Bpca = B * (coeff * coeff');

    % The average rloss on A and B of Vanilla PCA
    rloss_A(r) = rloss(A,approx_Apca,r);
    rloss_B(r) = rloss(B,approx_Bpca,r);
    rlosspca_max(r) = max(rloss_A(r),rloss_B(r));
    rloss_AoverBpca(r) = rloss_A(r)/rloss_B(r);

    % Fair PCA via eigenvalue Optimization
    tic
    U = fpca_Eigenvalue_Optimization(A, B, r,tol);
    time_Fair(r) = toc;

    % Projection of Fair PCA via eigenvalue Optimization
    approx_A = A * (U * U');
    approx_B = B * (U * U');

    % The average reconstruction rloss on A and B of Fair PCA via eigenvalue Optimization
    rlossFair_A(r) = rloss(A,approx_A,r);
    rlossFair_B(r) = rloss(B,approx_B,r);
    rlossFair_max(r) = max(rlossFair_A(r),rlossFair_B(r));
    rlossFair_AoverB(r) = rlossFair_A(r)/rlossFair_B(r);

    % Fair PCA via LP
    tic
    P_LP = fpca_LP(A,B,r,eta,T);
    time_FairLP(r) = toc;
    
    % Projection of Fair PCA via LP
    approxFair_A_LP = A * P_LP;
    approxFair_B_LP = B * P_LP;

    % The average reconstruction rloss on A and B of Fair PCA via LP
    rloss_A_LP(r) = rloss(A,approxFair_A_LP,r);
    rloss_B_LP(r) = rloss(B,approxFair_B_LP,r);
    rloss_LP_max(r) = max(rloss_A_LP(r),rloss_B_LP(r));
    rloss_AoverB_LP(r) = rloss_A_LP(r)/rloss_B_LP(r); 

    time_ratio(r) = time_FairLP(r)/time_Fair(r);
end

% Make a list for each reduced dimension
r_count = 1:r_total;
r_count = r_count';

% Table of different comparison result
% T = table(r_count,rlossFair_A,rlossFair_B,rlossFair_AoverB);
% T = table(r_count,rloss_A_LP,rloss_B_LP,rloss_AoverB_LP);
% T = table(r_count,rlossFair_max,rloss_LP_max);
% T = table(r_count,rloss_AoverBpca,rlossFair_AoverB,rloss_AoverB_LP);

% Plot the reconstruction rloss ratio figure
figure
x = r_count;
y1 = rlossFair_AoverB;
y2 = rloss_AoverB_LP;
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
legend("FPCA via eigenvalue optimization","FPCA via LP")
xlabel("Number of reduced dimensions")
ylabel("Reconstruction loss ratio")
title("Reconstruction loss Ratio")
% print -depsc newfigure1

% Plot the objective value figure
figure
x = r_count;
y1 = rlossFair_max;
y2 = rlosspca_max;
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
legend("FPCA via eigenvalue optimization","Vanilla PCA")
xlabel("Number of reduced dimensions")
ylabel("Objective value")
title("Objective value")
% print -depsc newfigure2


T = table(r_count,time_pca,time_Fair,time_FairLP,time_ratio)

% % Plot the efficiency figure
% figure
% x = r_count; 
% y1 = time_pca; plot(x,y1,"b--|",'LineWidth',3);
% hold on
% y2 = time_Fair; plot(x,y2,"r-s",'LineWidth',3);
% hold on
% y3 = time_FairLP; plot(x,y3,"g-o",'LineWidth',3);
% hold off
% legend("Vanilla PCA", "FPCA via eigenvalue optimization","FPCA via LP")
% xlabel("Number of reduced dimensions")
% ylabel("Runtime")
% title("Runtime")
% % print -depsc newfigure3
