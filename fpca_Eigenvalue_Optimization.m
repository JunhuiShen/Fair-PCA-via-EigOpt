function [U] = fpca_Eigenvalue_Optimization(XA, XB, r,tol)
% Fair PCA via convex optimization.
% Input: XA and XB are data matrices; r is the reduced dimension
% tol is tolerance
% Output: V is the solution to Fair PCA.

% Extract the dimension
d = size(XA,2);
na = size(XA,1);
nb = size(XB,1);

% Compute singular value of XA and XB
% sigval_a = svd(XA);
% sigval_b = svd(XB);
[~,Sa,~] = svds(XA,d,"largest",'Tolerance',tol);
sigval_a = diag(Sa);

[~,Sb,~] = svds(XB,d,"largest",'Tolerance',tol);
sigval_b = diag(Sb);

% Compute SA and SB
sa = sum(sigval_a(1:r).^2);
sb = sum(sigval_b(1:r).^2);

% Form HA and HB
Ha = (sa/r*eye(d) - XA'*XA)/na;
Hb = (sb/r*eye(d) - XB'*XB)/nb;

% Define H(t) and -phi(t)
H = @(t) t*Ha+(1-t)*Hb;
phiFun = @(t) -1 * sum(eigs(H(t), r, 'smallestreal')); % negative phi

% Find t0 that minimizes -phi(t)
myOpt.TolX = tol;
t0 = fminbnd(phiFun, 0, 1, myOpt);

% Return U where U contains the first r eigenvectors of H(t0)
[U,~] = eigs(H(t0), r, 'smallestreal');
end
