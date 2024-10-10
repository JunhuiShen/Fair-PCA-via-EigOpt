% demo for numerical range
close all; clear;
set(0, 'DefaultLegendInterpreter', 'latex');
set(0,'defaultTextInterpreter','latex');
rng(0);

n = 100;
d = 10;

na = 200; A = randn(na,n);
nb = 150; B = randn(nb,n);

sigval_a=svd(A);
sigval_b=svd(B);

sa = norm(sigval_a(1:d))^2;
sb = norm(sigval_b(1:d))^2;

Ha = (sa/d*eye(n) - A'*A)/na;
Hb = (sb/d*eye(n) - B'*B)/nb;


% generate boundary numerical range
nsample = 300;
THETA = linspace(0,2*pi,nsample); 
X =[]; Y =[];
for i = 1:nsample
	theta=THETA(i);
	HH = Ha*cos(theta) + Hb*sin(theta);
	[VV, EE] = eig(HH);
	ee = diag(EE); [~,idx] = sort(real(ee),'ascend')
	V = VV(:,idx(1:d));
	xx = trace(V'*Ha*V);
	yy = trace(V'*Hb*V);
	X = [X;xx];
	Y = [Y;yy];
end
figure; 
fill(X,Y,'y');


% plot contour of max function, mark intersection (approximately)
mx = max([X]); my = max([Y]);
text(mx/2, my*2/3, '$\mathcal W_r$', 'Fontsize', 15)

hold on; plot([0,mx],[0,mx],'linewidth', 2)

nsq = floor(nsample/4);
dxy = X(1:nsq)-Y(1:nsq); idx = find(dxy>0, 1, 'first');
P = polyfit(X(idx-1:idx), Y(idx-1:idx), 1);
ss = P(2)/(1-P(1));
plot(ss,ss, 'p', 'MarkerSize', 12, 'MarkerFaceColor', 'k')

dx=linspace(0,mx,100); dy=linspace(0,my,100);
[DX,DY] = meshgrid(dx,dy);
contour(DX, DY, max(DX,DY), '--k');


% 
xlim([0,mx+0.5]); ylim([0,my+0.5])
xlabel('$y_1$', 'fontsize', 20);
ylabel('$y_2$', 'fontsize', 20, 'rotation', 0);
axis square
