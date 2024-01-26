function [loss] = loss(Y, Z, d)
%original definition of trace
% loss of matrix Y with respect to matrix Z
% Y and Z are of the same size

Yhat = optApprox(Y, d);
loss =  re(Y, Z) - re (Y, Yhat);

function [Mhat] = optApprox(M, d)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
coeff = pca(M, 'NumComponents', d);
P = coeff * transpose(coeff);
Mhat = M*P;
end

function [reVal] = re(Y,Z)
% Calculate the reconstruction error of matrix Y with respect to matrix Z
% Matrix Y and Z are of the same size
reVal = norm(Y-Z, 'fro')^2;
end

end



