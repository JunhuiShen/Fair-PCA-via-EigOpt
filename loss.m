function [loss] = loss(Y, Z, d)
% loss of matrix Y with respect to matrix Z
% Y and Z are of the same size
coeff = pca(Y, 'NumComponents', d);
P = coeff * transpose(coeff);
Yhat = Y * P;
loss = (norm(Y - Z,"fro")^2 - norm(Y-Yhat,"fro")^2)/size(Y,1);
end



