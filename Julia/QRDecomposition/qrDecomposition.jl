#qrDecomposition.jl

module QRDecomposition

using LinearAlgebra

export qrGramSchmidt, absDet, qrSolve, qrHouseholder, hess

function qrGramSchmidt(A::Matrix{<:Real})
    """ This function computes the reduced QR decomposition of a real valued matrix A 
        via Modified Gram-Schmidt.
        Note: The matrix A (m x n) has to have rank n which is less than or equal to m.
    """
    # store dimensions of A
    m, n = size(A)
    # copy A into Q
    Q = float.(copy(A))
    # initialize R
    R = zeros(Number, (n,n))

    for i=1:n
        # diagonals will be set as the norm of the ith column of Q or A.
        R[i, i] = norm(Q[1:end, i], 2)
        # normalize the columns of Q
        Q[1:end, i] = Q[1:end, i]/R[i, i]
        for j=i+1:n
            # make the upper triangular elemnts of R.
            R[i, j] = dot(Q[1:end, j], Q[1:end, i])
            # Orthogonalize the jth column of Q.
            Q[1:end, j] -= (R[i, j] * Q[1:end, i])
        end
    end
    return Q, R
end

function qrGramSchmidt(A::Matrix{<:Complex})
    """ This function computes the reduced QR decomposition of a complex valued matrix A 
        via Modified Gram-Schmidt.
        Note: The matrix A (m x n) has to have rank n which is less than or equal to m.
    """
    # store dimensions of A
    m, n = size(A)
    # copy A
    Q = copy(A)
    # initialize R
    R = zeros(Complex{Float64}, (n, n))
    for i=1:n
        # diagonals will be set as the norm of the ith column of Q or A.
        R[i, i] = norm(Q[1:end, i], 2)
        # normalize the columns of Q
        Q[1:end, i] /= R[i, i] 
        for j=i+1:n
            # #make the upper triangular elemnts of R.
            R[i, j] = transpose(conj.(Q[1:end, i])) * Q[1:end, j]
            # Orthogonalize the jth column of Q.
            Q[1:end, j] -= (R[i, j] * Q[1:end, i])
        end
    end
    return Q, R
end

function absDet(A)
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A. A needs to be invertible
    """

    return abs(prod(diag(qr(A).R)))
end

function qrSolve(A::Matrix{<:Real}, b::Vector{<:Real})
    """Use the QR decomposition to efficiently solve the system Ax = b.
       We are assume that the matrix A is n x n and invertible
    """

    # caclulate QR decomposition
    F = qr(A)
    Q = F.Q
    R = F.R

    # calcualte matrix vector product
    y = transpose(Q) * b

    # now back subsittue to sovle for x
    n = size(A, 1)
    x = zeros(Float64, (n,))
    for i=n:-1:1 # start from the bottom and go to the top
        x[i] = y[i] # set initial element
        x[i] -= R[i, i+1:end]' * x[i+1:end] #subtracting off previous x elements
        x[i] /= R[i, i] # isolating x[i]
    end

    return x
end

function qrSolve(A::Matrix{<:Complex}, b::Vector{<:Complex})
    """Use the QR decomposition to efficiently solve the system Ax = b.
       We are assume that the matrix A is n x n and invertible
    """

    # caclulate QR decomposition
    F = qr(A)
    Q = F.Q
    R = F.R

    # calcualte matrix vector product
    y = conj.(transpose(Q)) * b

    # now back subsittue to sovle for x
    n = size(A, 1)
    x = zeros(Complex{Float64}, (n,))
    for i=n:-1:1 # start from the bottom and go to the top
        x[i] = y[i] # set initial element
        x[i] -= transpose(R[i, i+1:end]) * x[i+1:end] #subtracting off previous x elements
        x[i] /= R[i, i] # isolating x[i]
    end

    return x
end

# sign extraction functions
mySign(x::Real) = x >= 0 ? 1 : -1
mySign(x::Complex) = real(x) >= 0 ? 1 : -1

function qrHouseholder(A::Matrix{<:Real})
    m, n = size(A) # get shape of A
    R = copy(A) # copy A
    Q = Matrix{Float64}(I(m)) # Initialize Q as identity matrix.

    for k=1:n
        x = copy(R[k:end, k]) #initialize u
        e1 = zeros(Complex{Float64}, size(x))
        e1[begin] = 1+0im
        u = mySign(x[begin]) * norm(x, 2) .* e1 + x
        u /= norm(u, 2) # normalize u
        R[k:end, k:end] -= 2u * (conj(transpose(u)) * R[k:end, k:end]) # reflect R
        Q[k:end, 1:end] -= 2u * (conj(transpose(u)) * Q[k:end, 1:end]) # reflect Q
    end
    return transpose(Q), R
end

function qrHouseholder(A::Matrix{Complex{T}}) where T<:AbstractFloat
    # get shape of A
    m, n = size(A)
    # copy A to initalize R
    R = copy(A)
    # Initialize Q as the identity matrix
    Q = Matrix{Complex{T}}(I, m, m)

    for k = 1:n
        x = R[k:m, k]
        u = copy(x)
        # normalize v
        u[1] += mySign(u[begin])*norm(u, 2)
        u /= norm(u, 2) 

        R[k:m, k:n] -= 2u*(conj(transpose(u))*R[k:m, k:n])

        if k <= m
            Q[1:end, k:end] -= 2*Q[1:end, k:end]*u*u'
        end
    end
    return Q', R
end


# function qrHouseholder(A::Matrix{<:Complex})
#     m, n = size(A) # get shape of A
#     R = copy(A) # copy A
#     Q = Matrix{Complex{Float64}}(I(m)) # Initialize Q as identity matrix.

#     for k=1:n
#         u = copy(R[k:end, k]) #initialize u
#         u[1] += mySign(u[1]) * norm(u, 2) # reassign the first element
#         u /= norm(u, 2) # normalize u
#         R[k:end, k:end] -= 2u * (u' * R[k:end, k:end]) # reflect R
#         Q[k:end, 1:end] -= 2u * (u' * Q[k:end, 1:end]) # reflect Q
#     end
#     return transpose(Q), R
# end

function hess(A)
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T. A has to be nonsingular
    """
    # get shape of A
    m, n = size(A)
    # initialize H as A
    H = copy(A)
    # initialize Q as the identity matrix
    Q = Matrix{Float64}(I(m))

    for k=1:n-2
        # initalize u
        u = copy(H[k+1:end, k])
        u[1] += mySign(u[1]) * norm(u, 2) 
        # normalize u
        u /= norm(u, 2)
        # apply Qk to H
        H[k+1:end, k:end] -= 2u *(u'* H[k+1:end, k:end])
        # apply Qk^T to H
        H[1:end, k+1:end] -= 2H[1:end, k+1:end] * u * u'
        # apply Qk to Q
        Q[k+1:end, 1:end] -= 2u * (u' *  Q[k+1:end, 1:end])
    end

    return H, transpose(Q)
end

function givensTrinagularization(A::Matrix{T}) where T <: Real
    m, n= size(A)
    R = copy(A)
    Q = Matrix{Real{T}}(I, m, m)

    for j=1:n
        for i=m:-1:j+1
            a = R[i-1, j]
            b = R[i, j]
            G = [a b; -b a] / sqrt(a^2 + b^2) 
            R[i-1:i+1, j:end] *= G
            Q[i-1:i+1, 1:end] *= G
        end
    end

    Q', R
end

function givensTrinagularizationHessenberg(H::Matrix{T}) where T <: Real

    m, n = size(H)
    R = copy(H)
    Q = Matrix{Real{T}}(I, m, m)

    ub = min(n, m)

    for j=1:ub
        i = j + 1
        a = R[i-1, j]
        b = R[i, j]
        G = [a b; -b a] / sqrt(a^2 + b^2) 
        R[i-1:i+1, j:end] *= G
        Q[i-1:i+1, 1:end] *= G
    end
    return Q', R
end

end
