# lstsqEigs.jl

module LeastSquaresEigenvalues

using LinearAlgebra
using Random
export leastSquareSol
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

# function test()

#     d = MersenneTwister(0)
#     A = rand(d, Float64, (6, 4))
#     b = rand(d, Float64, (6,))

#     x = leastSquareSol(A, b)

#     y = A\b

#     @assert x ≈ y
    
# end

# test()



end
