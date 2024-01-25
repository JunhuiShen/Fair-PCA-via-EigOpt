function [V,Ha,Hb] = fpca(A, B, d, tol)
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
sigval_a=svd(A);
sigval_b=svd(B);
sa = norm(sigval_a(1:d))^2;
sb = norm(sigval_b(1:d))^2;
Ha = (sa/d*eye(n) - A'*A)/na;
Hb = (sb/d*eye(n) - B'*B)/nb;

% Eigopt
H = @(t) t*Ha+(1-t)*Hb;
phiFun = @(t) -1*sum(eigs(H(t), d, 'smallestreal'));    % negative phi(t)

myOpt.TolX = tol;
t0 = fminbnd(phiFun, 0, 1, myOpt);	% univariate optimization
[V,~] = eigs(H(t0), d, 'smallestreal');
end