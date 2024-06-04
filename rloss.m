function [rloss] = rloss(M, Mapprox, r)
% Compute the average reconstruction loss
% where Mbestapprox is the best rank-r approximation of M
U = pca(M, 'NumComponents', r);
Mbestapprox = M * (U * U');
rloss = (norm(M - Mapprox,"fro")^2 - norm(M-Mbestapprox,"fro")^2)/size(M,1);
end




