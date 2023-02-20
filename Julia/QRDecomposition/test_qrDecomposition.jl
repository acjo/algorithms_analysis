# test_qrDecomposition.jl
include("qrDecomposition.jl")
using .QRDecomposition
using LinearAlgebra, Random

function test()

    A = [1.0 -2.0 3.5; 
        1.0 3.0 -0.5; 
        1.0 3.0 2.5; 
        1.0 -2.0 0.5]

    Q, R = qrGramSchmidt(A)

    @assert Q*R ≈ A
    @assert UpperTriangular(R) ≈ R
    @assert transpose(Q) * Q ≈ I(3)

    F = qr(A)

    @assert Matrix(F.Q) ≈ -Q
    @assert F.R ≈ -R

    A = rand(Float64, (10,10))
    determinantAbs = absDet(A)

    @assert abs(det(A)) ≈ determinantAbs

    
    d = MersenneTwister(0)
    A = rand(d, Float64, (10,10))
    b = rand(d, Float64, (10,))
    sol = A\b
    qrSol = qrSolve(A, b)

    @assert sol ≈ qrSol

    A = rand(Float64, (4,3))
    Q, R = qrHouseholder(A)
    # println(size(Q))
    # println(size(R))
    @assert Q*R ≈ A
    @assert transpose(Q) * Q ≈ I(4)

    F = LinearAlgebra.QRCompactWY
    F = qr(A)

    @assert F.Q ≈ Q
    @assert triu(F.factors) ≈ R

    A = rand(Float64, (8,8))

    H, Q = hess(A)
    @assert Q * H * transpose(Q) ≈ A 
    @assert triu(H, -1) ≈ H
    F = hessenberg(A)

    @assert F.H ≈ H
    @assert F.Q ≈ Q

    println("All Tests Passed")

    return 
end

test()
