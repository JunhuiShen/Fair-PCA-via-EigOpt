function [rloss] = rloss(Y, Z, r)
% Compute ||Y - Z||_{F}^{2} - ||Y - Yhat||_{F}^{2}
% where Yhat is the best rank-r approximation of Y
coeff = pca(Y, 'NumComponents', r);
Yhat = Y * (coeff * transpose(coeff));
rloss = (norm(Y - Z,"fro")^2 - norm(Y-Yhat,"fro")^2)/size(Y,1);
end




