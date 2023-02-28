# lstsqEigs.jl

module LeastSquaresEigenvalues

using LinearAlgebra
using Random
export leastSquareSol, powerMethod, qrAlgorithm, inversePowerMethod

function leastSquareSol(A, b)
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) Matrix): A matrix of rank n <= m.
        b ((m, ) Vector): A vector of length m.

    Returns:
        x ((n, ) Vector): The solution to the normal equations.
    """

    F = qr(A)
    tempVec = transpose(Matrix(F.Q)) * b
    return F.R\tempVec 
end


function powerMethod(A::Matrix{<:Real}; N::Int=20, tol::Float64=1e-12)
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) Matrix): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) Vector): An eigenvector corresponding to the dominant
        eigenvalue of A.
    """
    # get matrix dimensions
    m, n = size(A) 
    # normalize random vector
    x0 = rand(Float64, (n,))
    x0 /= norm(x0, 2)
    x1 = copy(x0)

    for k=1:N
        x1 = A * x0
        x1 /= norm(x1, 2)
        if norm(x1 - x0) < tol
            break
        end
        x0 = copy(x1)
    end
    return x1' * A * x1, x1
end

function qrAlgorithm(A::Matrix{<:Real}; N::Int=20, tol::Float64=1e-12)
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) Matrix): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
        block is 1x1 or 2x2.

    Returns:
        ((n,) Vector): The eigenvalues of A.
    """
    m, n = size(A)
    Hess = hessenberg(A)
    S = Hess.H

    # while 0 <= K < n iteratively copute Q, R, and S = RQ
    for k=1:N
        QR = qr(S)
        S = QR.R * QR.Q
    end

    eigs= []
    i = 1
    while i <= n # iterate through s 
        # if true S_i is a 1x1 matrix
        if S[i, i] == S[n, n] || abs(S[i+1, i]) < tol
            append!(eigs, S[i, i])
        else
            # use the quadratic formula toc ompute the eigenvalues
            discriminant = √(Complex(((S[i, i] + S[i+1, i+1])^2  - 4 * (S[i, i] * S[i+1, i+1] - S[i, i+1]*S[i+1, i]))))
            eig1 = (S[i, i] + S[i+1, i+1] + discriminant) / 2
            eig2 = (S[i, i] + S[i+1, i+1] - discriminant) / 2
            append!(eigs, eig2)
            append!(eigs, eig1)
            i += 1
        end
        i += 1
    end
    return eigs
end

function inversePowerMethod(A::Matrix{Vector{T}}, μ::T) where T <: Number

    m,n = size(A)
    x0 = rand(T, (n,))
    x0 /= norm(x0, 2)
    F = qr(A)

    Q = F.Q
    R = F.R

    for k=1:n
        # solve system
        x1 = R\(Q'*x0)
        # take then orm
        x1 /= norm(x1, 2)
        # copy for next iteration
        x0 = copy(x1)

    return x0' * A * x0, x0

end


end
