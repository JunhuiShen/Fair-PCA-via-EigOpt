function U = FPCAviaEigOpt(A, B, r, tol)
    % Fair PCA via Eigenvalue Optimization
    %
    % Inputs:
    %   A   - data matrix for group A
    %   B   - data matrix for group B 
    %   r   - reduced dimension (number of principal components)
    %   tol - tolerance for the optimization algorithm
    %
    % Outputs:
    %   U - the solution to Fair PCA

    % Extract dimensions
    d = size(A, 2); % Number of features (columns) in A and B
    na = size(A, 1); % Number of samples (rows) in A
    nb = size(B, 1); % Number of samples (rows) in B
    
    % Compute the singular values of A and B 
    [~, sa, ~] = svds(A, d, "largest", 'Tolerance', tol);
    sigval_a = diag(sa); % Singular values of A

    [~, sb, ~] = svds(B, d, "largest", 'Tolerance', tol);
    sigval_b = diag(sb); % Singular values of B
    
    % Compute SA and SB as the sum of squares of the largest r singular values
    SA = sum(sigval_a(1:r) .^ 2);
    SB = sum(sigval_b(1:r) .^ 2);

    % Form HA and HB matrices
    HA = (SA / r * eye(d) - A' * A) / na;
    HB = (SB / r * eye(d) - B' * B) / nb;

    % Define H(t) as a convex combination of HA and HB
    H = @(t) t * HA + (1 - t) * HB;

    % Define phiFun as the sum of the smallest r eigenvalues of H(t)
    phiFun = @(t) sum(eigs(H(t), r, 'smallestreal', 'Tolerance', tol));

    % Brent's method
    options = optimset('TolX', tol);
    t_star = fminbnd(@(t) -phiFun(t), 0, 1, options);

    % Compute the first r eigenvectors of H(t_star)
    [U, ~] = eigs(H(t_star), r, 'smallestreal');

    % % Plot of phi(t)
    % figure;
    % fplot(phiFun, [-0.2, 1.2], 'LineWidth', 5);
    % 
    % % Set x-axis limits from -0.2 to 1.2
    % xlim([-0.2, 1.2]);
    % 
    % % Add dashed vertical lines at x = 0 and x = 1
    % hold on;
    % plot([0, 0], ylim, '--k', 'LineWidth', 1.2); 
    % plot([1, 1], ylim, '--k', 'LineWidth', 1.2);      
    % 
    % % Set the text interpreter to LaTeX
    % xlabel('t', 'Interpreter', 'latex','FontSize',30);
    % ylabel('$\phi(t)$', 'Interpreter', 'latex', 'Rotation', 0,'FontSize',30);
    % 
    % % Turn off grid
    % grid off;
    % hold off;
end