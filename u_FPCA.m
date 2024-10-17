%% This function finds the minimum of a unimodal function
% We consider the golden section search algorithm

function proj_matrix = u_FPCA(A,B,r,tol)

     % Input: 
    % A - Data matrix for group A (na samples, m features)
    % B - Data matrix for group B (nb samples, m features)
    % r - Target reduced dimension
    % tol - Tolerance for golden section search
    
    % Sizes
    na = size(A, 1);  % Number of samples in group A
    nb = size(B, 1);  % Number of samples in group B
    n = na + nb;      % Total number of samples
    m = size(A, 2);   % Number of features
    
    % Combine A and B for PCA
    M = [A; B];
    
    % Perform standard PCA on combined data
    coeff = pca(M);  % Eigenvector coefficients from PCA
    covM = (M' * M) / n;  % Covariance matrix of M
    
    % Determine the covariance difference between groups A and B
    covA = (A' * A) / na;
    covB = (B' * B) / nb;
    dif_cov = covB - covA;  % Covariance difference
    
    % Initialize parameters for golden section search
    alpha0 = 0;
    alpha1 = 1;

    % Projection matrix (PCA)
    proj_pca = coeff(:,1:r)*coeff(:,1:r)';
    rec_pca = re(M,M*proj_pca)/n;
    recA_pca = re(A,A*proj_pca)/na;
    recB_pca = re(B,B*proj_pca)/nb;
    rec_difs_pca = (recB_pca - recA_pca)^2; % Calculate the squared difference
    
    % Defining the privileged group
    if recA_pca <= recB_pca
        dif_cov = (B'*B)/nb - (A'*A)/na;
    else
        dif_cov = (A'*A)/na - (B'*B)/nb;
    end

    proj_matrix = golden_section_function(alpha0,alpha1,covM,dif_cov,M,A,B,m,n,na,nb,r,max(recA_pca,recB_pca));
    

%% This function finds the minimum of a unimodal function
% We consider the golden section search algorithm

function proj_matrix = golden_section_function(alpha0,alpha1,covM,dif_cov,M,A,B,m,n,na,nb,r,maxRec)

    % Default parameters
    
    g_ratio = (sqrt(5) + 1)/2; % Golden ratio

    % Weighted covariance matrices
    X_0 = alpha0*covM + (1-alpha0)*(dif_cov);
    X_1 = alpha1*covM + (1-alpha1)*(dif_cov);

    % Eigenvector/value decomposition
    [V_0,D_0] = eig(X_0); V_0 = V_0(:,m+1-[1:m]); d_aux = diag(D_0); D_0(1:m+1:end) = d_aux(m+1-[1:m]);
    [V_1,D_1] = eig(X_1); V_1 = V_1(:,m+1-[1:m]); d_aux = diag(D_1); D_1(1:m+1:end) = d_aux(m+1-[1:m]);

    % Projection matrix (proposal)
    proj_0 = V_0(:,1:r)*V_0(:,1:r)';
    proj_1 = V_1(:,1:r)*V_1(:,1:r)';

    % Reconstruction errors for the proposed approach
    recA_0 = re(A,A*proj_0)/na;
    recB_0 = re(B,B*proj_0)/nb;
    rec_0 = re(M,M*proj_0)/n;
    rec_difs_0 = (recB_0 - recA_0)^2;

    recA_1 = re(A,A*proj_1)/na;
    recB_1 = re(B,B*proj_1)/nb;
    rec_1 = re(M,M*proj_1)/n;
    rec_difs_1 = (recB_1 - recA_1)^2;

    while abs(alpha1-alpha0) > tol
    
        % Novel points
        alpha0_aux = alpha1 - (alpha1-alpha0)/g_ratio;
        alpha1_aux = alpha0 + (alpha1-alpha0)/g_ratio;
    
        % Weighted covariance matrices
        X_0 = alpha0_aux*covM + (1-alpha0_aux)*(dif_cov);
        X_1 = alpha1_aux*covM + (1-alpha1_aux)*(dif_cov);
    
        % Eigenvector/value decomposition
        [V_0,D_0] = eig(X_0); V_0 = V_0(:,m+1-[1:m]); d_aux = diag(D_0); D_0(1:m+1:end) = d_aux(m+1-[1:m]);
        [V_1,D_1] = eig(X_1); V_1 = V_1(:,m+1-[1:m]); d_aux = diag(D_1); D_1(1:m+1:end) = d_aux(m+1-[1:m]);
    
        % Projection matrix (proposal)
        proj_0 = V_0(:,1:r)*V_0(:,1:r)';
        proj_1 = V_1(:,1:r)*V_1(:,1:r)';
    
        % Reconstruction erros for the proposed approach
        recA_0 = re(A,A*proj_0)/na;
        recB_0 = re(B,B*proj_0)/nb;
        rec_0 = re(M,M*proj_0)/n;
        rec_difs_0 = (recB_0 - recA_0)^2;
    
        recA_1 = re(A,A*proj_1)/na;
        recB_1 = re(B,B*proj_1)/nb;
        rec_1 = re(M,M*proj_1)/n;
        rec_difs_1 = (recB_1 - recA_1)^2;
    
        if rec_difs_0 < rec_difs_1
            alpha1 = alpha1_aux;
        else
            alpha0 = alpha0_aux;
        end
    end

    alpha = (alpha1 + alpha0)/2;

    % Weighted covariance matrices
    X = alpha*covM + (1-alpha)*(dif_cov);

    % Eigenvector/value decomposition
    [V,D] = eig(X); V = V(:,m+1-[1:m]); d_aux = diag(D); D(1:m+1:end) = d_aux(m+1-[1:m]);

    % Projection matrix (proposal)
    proj_matrix = V(:,1:r)*V(:,1:r)';

    end


% Nested function to calculate reconstruction differences
    function rec_difs = calculate_rec_difs(alpha, covM, dif_cov, A, B, na, nb, r)
        % Calculate the difference in reconstruction errors for a given alpha
        X = alpha * covM + (1 - alpha) * dif_cov;
        [V, ~] = eig(X);
        proj = V(:, 1:r) * V(:, 1:r)';  % Projection matrix

        % Reconstruction errors for group A and B
        recA = re(A, A * proj) / na;
        recB = re(B, B * proj) / nb;
        
        % Squared difference in reconstruction errors
        rec_difs = (recB - recA)^2;
    end

    % Nested function to calculate the reconstruction error
    function reVal = re(Y, Z)
        % Calculate the reconstruction error between matrices Y and Z
        reVal = norm(Y - Z, 'fro')^2;
    end

end

