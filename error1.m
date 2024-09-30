function err = error1(A, B)
    % frobenius_error calculates the Frobenius norm squared error between matrices A and B
    %
    % Inputs:
    %   A - First matrix
    %   B - Second matrix
    %
    % Output:
    %   error - Frobenius norm squared error ||A - B||_F^2

    err = norm(A - B, 'fro')^2;
end