module LinearSystems

using LinearAlgebra, Random, Distributions, Plots, LaTeXStrings, SparseArrays
export ref

function ref(A)
    """This function reduces the square matrix A to REF.
    Parameters:
        A ((n,n) Matrix): The square invertible matrix to be reduced.
    Returns:
        ((n,n) Matrix): The REF of A.
    """

    A = float.(A) # change to type float so we can modify
    rows, cols = size(A) # get dimensions of A 

    for col=1:cols
        for row=col+1:rows
            if A[row, col] == 0
                continue
            else
                A[row, col:end] -= (A[row, col]/A[col, col]) * A[col, col:end]
            end
        end
    end

    return A

end

function myLU(A)
    """This function computes the LU decomposition of the square matrix A.
    Parameters:
        A ((n,n) Matrix): The matrix to decompose.
    Returns:
        L ((n,n) Matrix): The lower-triangular part of the decomposition.
        U ((n,n) Matrix): The upper-triangular part of the decomposition.
    """
    m, n = size(A)
    U = float.(copy(A))
    L = Matrix(1.0I, m, m)
    for jj=0:n
        j = jj + 1
        for ii=jj+1:m-1
            i = ii + 1
            if U[j, j] == 0
                continue
            end
            L[i, j] = U[i, j]/U[j, j]
            U[i,j:end] = U[i,j:end] - (L[i,j]*U[j, j:end])
        end
    end
    return L, U
end

function solve(A, b)
    """This function uses the LU decomposition and back substitution to solve Ax = b
    Parameters:
        A ((n,n) Matrix)
        b ((n,) Vector)
    Returns:
        x ((n,) Vector): The solution to the linear system.
    """

    L, U = myLU(A)
    rows, _ = size(A)
    y = zeros(rows)
    x = zeros(rows)

    for kk=1:rows
        y[kk] = b[kk]
        for jj=1:kk-1
            y[kk] -=  L[kk, jj]*y[jj]
        end
    end

    for kk=rows:-1:1
        x[kk] = y[kk]
        for jj=kk+1:rows
            x[kk] -= U[kk,jj]*x[jj]
        end
        x[kk] /= U[kk, kk]
    end

    return x
end

function problem4()
    """
    Time different functions in the LinearAlgebra library.
    """

    sizes = [2^i for i=1:12]
    timeLAInv = []
    timeLASolve = []
    timeLUFactor = []
    timeLUSolve = []
    
    for n in sizes
        println(n)
        b = rand(Float64, n)
        A = rand(Float64, (n,n))
        # time inverse
        t = @timed inv(A)*b
        append!(timeLAInv, t.time)

        # time la solve
        t = @timed A\b
        append!(timeLASolve, t.time)

        # time LU solve with factorization
        t = @timed lu(A)\b
        append!(timeLUFactor, t.time)

        # time LU solve without factorization 
        F = lu(A)
        t = @timed F\b
        append!(timeLUSolve, t.time)
    end

    plotlyjs()
    plt = plot(;xscale=:log2, yscale=:log2, migrogrid=true, dpi=1000)
    plot!(plt, sizes, timeLAInv; label="Inverse", lw=3)
    plot!(plt, sizes, timeLASolve; label="Solve", lw=3)
    plot!(plt, sizes, timeLUFactor; label="LU factorization", lw=3)
    plot!(plt, sizes, timeLUSolve; label="LU Solve", lw=3)
    title!(plt, "Timing linear solution algorithms")
    xlabel!(plt, L"Array size $(n\times n)$")
    ylabel!(plt, "Time (s)")

    display(plt)

    return plt
end

function problem5(n::Int)

    B = spzeros(n,n)
    B[diagind(B)] .= -4
    B[diagind(B,-1)] .= 1
    B[diagind(B, 1)] .= 1

    A = copy(B)
    for i in 1:n-1
        A = blockdiag(A, B)
    end

    A[diagind(A, -n)] .= 1
    A[diagind(A, n)] .= 1

    return A
end


function problem6()

    sparse_time = []
    dense_time = []
    dimension = []
    for n in 2 .^ (1:6)
        # generate sparse matrix
        ASparse = problem5(n)
        append!(dimension, size(ASparse, 1))
        # generate dense matrix
        ADense = Matrix(ASparse)
        # generate random vector
        b = rand(Float64, n^2)
        # solve system using sparse and dense operators
        tSparse = @timed x = ASparse\b
        tDense = @timed x = ADense\b
        append!(sparse_time, tSparse.time)
        append!(dense_time, tDense.time)
    end

    plt = plot(;xscale=:log2, yscale=:log2, migrogrid=true, dpi=300)
    plot!(dimension, sparse_time, label="Sparse Solver", lw=3)
    plot!(dimension, dense_time, label="Dense Solver", lw=3)
    title!(plt, "Sparse vs Dense array linear solve times")
    xlabel!(plt, "Dimension")
    ylabel!(plt, "Time (s)")
    display(plt)

    return
end

problem6()
end

