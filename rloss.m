function [rloss] = rloss(M, Mapprox, r)
    % rloss calculates the reconstruction loss of approximation quality
    %
    % Inputs:
    %   M        - Original data matrix
    %   Mapprox  - Approximated data matrix
    %   r        - Number of principal components
    %
    % Output:
    %   rloss - Reconstruction loss of the approximation

    % Perform PCA to get the first r principal components
    U = pca(M, 'NumComponents', r);

    % Compute the best rank-r approximation of M using the principal components
    Mbestapprox = M * (U * U');
    
    % The reconstruction loss is normalized by the number of rows in M
    rloss = (error1(M, Mapprox) - error1(M,Mbestapprox))/size(M, 1);
end





