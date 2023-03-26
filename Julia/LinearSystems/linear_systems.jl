module LinearSystems

using LinearAlgebra, Random, Distributions, Plots, LatexStrings
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

    return
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
    rows, cols = size(A)
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
            x[kk] -= u[kk,jj]*x[jj]
        end
        x[kk] /= u[kk, kk]
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
    plt = plot(sizes, timeLAInv; label="Inverse" )
    plot!(sizes, timeLASolve; label="Solve" )
    plot!(sizes, timeLUFactor; label="LU factorization")
    plot!(sizes, timeLUSolve; label="LU Solve" )
    plot!(scalex=:log2, scaley=:log2)
    title!("Timing linear solution algorithms")
    xlabel!(L"Array size $(n\times n)$")
    ylabel!("Time")
    display(plt)
    readline()
end

function test()

    problem4()
    # A = rand(Float64, (3,3))
    # b = rand(Float64, 3)

    # L, U = myLU(A)
    # F = lu(A)

    # println(F.L)
    # println("\n")
    # println(L)
    # @assert F.L ≈ L
    # @assert F.U ≈ U


    # x = solve(A, b)

    # juliaSol = A\b

    # # println(juliaSol)
    # @assert x ≈ juliaSol

end

test()


end