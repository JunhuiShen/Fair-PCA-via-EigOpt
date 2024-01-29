function [U] = fpca(A, B, r,tol)
% Fair PCA via eigenvalue optimization.
% Input: 
%   A, B are data matrices; 
%   d is reduced dimension
%   tol is tolerance for eigopt (e.g., 1.0E-8)
% Output:
%   V is solution to Fair PCA.
%

% Data matrices
n = size(A,2);
na = size(A,1);
nb = size(B,1);

[~,Sa,~] = svds(A,n,"largest",'Tolerance',tol);
sigval_a = diag(Sa);

[~,Sb,~] = svds(B,n,"largest",'Tolerance',tol);
sigval_b = diag(Sb);

sa = sum(sigval_a(1:r).^2);
sb = sum(sigval_b(1:r).^2);

Ha = (sa/r*eye(n) - A'*A)/na;
Hb = (sb/r*eye(n) - B'*B)/nb;

% Eigopt
H = @(t) t*Ha+(1-t)*Hb;
phiFun = @(t) -1*sum(eigs(H(t), r, 'smallestreal'));    % negative phi(t)

myOpt.TolX = tol;
t0 = fminbnd(phiFun, 0, 1, myOpt);	% univariate optimization
[U,~] = eigs(H(t0), r, 'smallestreal');
end
