function [loss] = loss(Y, Z, r)
% loss of matrix Y with respect to matrix Z
% Y and Z are of the same size
coeff = pca(Y, 'NumComponents', r);
Yhat = Y * (coeff * transpose(coeff));
loss = (norm(Y - Z,"fro")^2 - norm(Y-Yhat,"fro")^2)/size(Y,1);
end



