function U = FairPCAviaEigOpt(A, B, r, tol)
% Fair PCA via Eigenvalue Optimization
% Input: A and B are data matrices; r is the reduced dimension
% tol is tolerance
% Output: U is the solution to Fair PCA.

% Extract the dimension
d = size(A,2);
na = size(A,1);
nb = size(B,1);

% Compute singular value of A and B
[~,sa,~] = svds(A,d,"largest",'Tolerance',tol);
sigval_a = diag(sa);

[~,sb,~] = svds(B,d,"largest",'Tolerance',tol);
sigval_b = diag(sb);

% Compute SA and SB
SA = sum(sigval_a(1:r).^2);
SB = sum(sigval_b(1:r).^2);

% Form HA and HB
HA = (SA/r*eye(d) - A'*A)/na;
HB = (SB/r*eye(d) - B'*B)/nb;

% Define H(t) and -phi(t)
H = @(t) t*HA+(1-t)*HB;
phiFun = @(t) -1 * sum(eigs(H(t), r, 'smallestreal')); % negative phi

myOpt.TolX = tol;

% Find t0 that minimizes -phi(t)
[t0,~,~,output] = fminbnd(phiFun, 0, 1, myOpt);

% Analysis of fminbnd ()
% fprintf("Current r = %d \n",r);
% fprintf("Number of iterations: %d\n", output.iterations);

% Return U where U contains the first r eigenvectors of H(t0)
[U,~] = eigs(H(t0), r, 'smallestreal');
end
