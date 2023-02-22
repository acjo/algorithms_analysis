# test_qrDecomposition.jl
include("qrDecomposition.jl")
using .QRDecomposition
using LinearAlgebra, Random

function test()
    """
    Tests to verify functionality in the functions given in the QRDecomposition module
    """
    # random number generator
    d = MersenneTwister(0)
    ############################
    # Testing qr factorization using Gram-Schmidt
    # testing on real valued matrix
    ############################
    A = [1.0 -2.0 3.5; 
        1.0 3.0 -0.5; 
        1.0 3.0 2.5; 
        1.0 -2.0 0.5]
    Q, R = qrGramSchmidt(A)
    F = qr(A)
    @assert Q*R ≈ A # verify product
    @assert UpperTriangular(R) ≈ R # verify R is upper triangular
    @assert transpose(Q) * Q ≈ I(3) # verify Q is orthonormal
    @assert Matrix(F.Q) ≈ -Q # verify algorithm matches Julia's
    @assert F.R ≈ -R # verify algorithm matches Julia's

    ############################
    # Testing qr factorization using Gram-Schmidt
    # testing on complex valued matrix
    ############################
    A = rand(d, Complex{Float64}, (4,3))
    Q, R = qrGramSchmidt(A)
    F = qr(A)
    @assert Q*R ≈ A
    @assert UpperTriangular(R) ≈ R
    @assert conj.(transpose(Q)) * Q ≈ I(3)
    @assert Matrix(F.Q) ≈ -Q
    @assert F.R ≈ -R

    ############################
    # testing determinant calculation 
    # Testing on real valued matrix
    ############################
    A = rand(d, Float64, (10,10))
    determinantAbs = absDet(A)
    @assert abs(det(A)) ≈ determinantAbs
    ############################
    # testing determinant calculation 
    # Testing on complex valued matrix
    ############################
    A = rand(d, Complex{Float64}, (10,10))
    determinantAbs = absDet(A)
    @assert abs(det(A)) ≈ determinantAbs

    ############################
    # testing solver using QR factorization
    # Testing on real valued matrix
    ############################
    A = rand(d, Float64, (10,10))
    b = rand(d, Float64, (10,))
    sol = A\b
    qrSol = qrSolve(A, b)
    @assert sol ≈ qrSol

    ############################
    # testing solver using QR factorization
    # Testing on complex valued matrix
    ############################
    A = rand(d, Complex{Float64}, (10,10))
    b = rand(d, Complex{Float64}, (10,))
    sol = A\b
    qrSol = qrSolve(A, b)
    @assert sol ≈ qrSol

    ############################
    # testing QR factorization using Householder reflections
    # Testing on real valued matrix
    ############################
    A = rand(d, Float64, (4,3))
    Q, R = qrHouseholder(A)
    F = qr(A)
    @assert Q*R ≈ A
    @assert transpose(Q) * Q ≈ I(4)
    @assert F.Q ≈ Q
    @assert triu(F.factors) ≈ R

    ############################
    # testing QR factorization using Householder reflections
    # Testing on complex valued matrix
    ############################
    A = rand(d, Complex{Float64}, (4,3))
    Q, R = qrHouseholder(A)
    F = qr(A)
    println("\nJulia: \n")
    Base.print_matrix(stdout, triu(F.factors))
    println("\nMine: \n")
    Base.print_matrix(stdout, R)
    println("\n\n")
    @assert Q*R ≈ A
    @assert transpose(Q) * Q ≈ I(4)
    @assert F.Q ≈ Q
    @assert triu(F.factors) ≈ R

    ############################
    # testing Hessenberg form 
    # Testing on Real valued matrix
    ############################
    A = rand(d, Float64, (8,8))
    H, Q = hess(A)
    F = hessenberg(A)
    @assert Q * H * transpose(Q) ≈ A 
    @assert triu(H, -1) ≈ H
    @assert F.H ≈ H
    @assert F.Q ≈ Q

    println("All Tests Passed")

    return 
end

test()