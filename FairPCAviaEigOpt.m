function [U, t_vals] = FairPCAviaEigOpt(A, B, r, tol)
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
    %   t_vals - trajectory of points chosen by Brent's method

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

    % Brent's method to find the trajectory of t values
    t_vals = [];
    options = optimset('TolX', tol);
    % options = optimset('TolX', tol, 'OutputFcn', @(t, optimValues, state) outfun(t, optimValues, state));
    [t_star, ~] = fminbnd(@(t) -phiFun(t), 0, 1, options);

    % Compute the first r eigenvectors of H(t_star)
    [U, ~] = eigs(H(t_star), r, 'smallestreal');
    
    % % Helper function to capture Brent's method iterates
    % function stop = outfun(t, optimValues, state)
    %     stop = false;
    %     if isequal(state, 'iter')
    %         t_vals = [t_vals, t]; % Store t values at each iteration
    %     end
    % end
    % 
    % % Plot phi(t) and mark the iterates t*
    % figure;
    % 
    % % Main plot of phi(t) over [0, 1]
    % fplot(phiFun, [0, 1], 'LineWidth', 1.5);
    % hold on;
    % 
    % % Evaluate phi(t) at each iterate t*
    % phi_vals = arrayfun(phiFun, t_vals); 
    % plot(t_vals, phi_vals, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    % 
    % % Annotate iterates with numbers
    % for i = 1:length(t_vals)
    %     text(t_vals(i), phi_vals(i) + 0.02, sprintf('%d', i), 'FontSize', 8, 'FontWeight', 'bold', ...
    %          'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'black');
    % end
    % 
    % % Labels and legend
    % xlabel("t");
    % ylabel("");
    % legend('\phi(t)', 'Iterates t*', 'Location', 'best');
    % grid on;
    % 
    % % Create zoomed-in inset (subplot)
    % ax2 = axes('Position', [0.4, 0.2, 0.25, 0.25]); % Adjusted inset position
    % box on;
    % 
    % % Zoom into the middle iterates (adjust x and y limits based on t_vals range)
    % middle_idx = floor(length(t_vals) / 2);
    % t_range = [t_vals(middle_idx) - 0.1, t_vals(middle_idx) + 0.1];
    % 
    % % Plot phi(t) in zoomed-in region
    % fplot(ax2, phiFun, t_range, 'LineWidth', 1.5);
    % hold on;
    % 
    % % Plot the iterates in the zoomed-in region
    % plot(ax2, t_vals, phi_vals, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    % 
    % % Set x and y limits for the zoomed plot
    % set(ax2, 'XLim', t_range, 'YLim', [min(phi_vals) - 0.01, max(phi_vals) + 0.01]);
    % 
    % % Labels for zoomed-in subplot (remove ylabel)
    % xlabel(ax2, 'Iterate t_*');
    % ylabel(ax2, '');
    % 
    % hold off;

    % % Plot phi(t) and mark the iterates t*
    % figure;
    % 
    % % Main plot of phi(t) over [0, 1]
    % fplot(phiFun, [0, 1], 'LineWidth', 1.5);
    % 
    % grid off; 
end
