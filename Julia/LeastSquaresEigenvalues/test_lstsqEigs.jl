# test_lstsqEigs.jl

include("lstsqEigs.jl")
using .LeastSquaresEigenvalues 
using LinearAlgebra
using Random
using Polynomials
using NPZ
using Plots

function test()

    d = MersenneTwister(0)
    A = rand(d, Float64, (6, 4))
    b = rand(d, Float64, (6,))

    x = leastSquareSol(A, b)

    y = A\b

    @assert x ≈ y
    

    A = rand(d, Float64, (10,10))

    largestEValue, vec = powerMethod(A; N=50)
    E = eigen(A)

    @assert E.values[end] ≈ largestEValue
    @assert E.vectors[:, end] ≈ -vec

    EValues = qrAlgorithm(A; N=1000)

    myRealPart = abs.(EValues)
    juliaRealPart =abs.(E.values)

    myPermutation = sortperm(myRealPart)
    juliaPermutation = sortperm(juliaRealPart)

    println("All Tests Passed!")

    return
end

function lineFit()
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """

    Data = npzread("housing.npy") 
    years = Data[1:end, 1] #get x axis data
    prices = Data[1:end, 2] # get y axis data

    # get the least squares solution

    A = [years ones(length(years))]

    # least squares solution
    solution = A\prices
    a = solution[1]
    b = solution[2]


    # defines the function we will use to plot the solution
    line(x) = a .* x .+ b
    plt = plot(years, line(years), label="Line of Best Fit" )
    scatter!(years, prices)
    xlabel!("Year (+2000)")
    ylabel!("Prices")
    title!("Housing Market")
    display(plt)
    return
end

function buildVandermonde(x::Vector{<:Real}, p::Int64)
    """
    Uses the evaluation points in x to create the vandermonde matrix 
    for finding the coefficients of the best fit polynomial of the 
    points of degree p. 
    Parameters:
        x ((N,) Vector): Nx1 Vector of evaluation points
        p (int): the degree of the fitting polynomial 
    Returns:
        V ((N, p+1) Matrix): Vandermond matrix for finding the degree 
        p polynomial.
    """

    V = ones(Float64, (size(x, 1), p+1))

    for i=1:size(x,1)
        for n=0:p-1
            V[i, n+1] = x[i]^(p-n)
        end
    end
    return V
end

function polynomialFit()
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    Data = npzread("housing.npy") 
    years = Data[1:end, 1] #get x axis data
    prices = Data[1:end, 2] # get y axis data

    # create Vandermonde matrices
    A3 = buildVandermonde(years, 3)
    A6 = buildVandermonde(years, 6)
    A9 = buildVandermonde(years, 9)
    A12 = buildVandermonde(years, 12)

    # solve the least squares problems
    y3 = A3\prices
    y6 = A6\prices
    y9 = A9\prices
    y12 = A12\prices

    # construct polynomials
    p3 = Polynomial(y3[end:-1:begin]) # degree 3
    p6 = Polynomial(y6[end:-1:begin]) # degree 6
    p9 = Polynomial(y9[end:-1:begin]) # degree 9
    p12 = Polynomial(y12[end:-1:begin]) # degree 12
    jP13 = fit(years, prices, 13) # degree 13 from polynomial package

    # setting domain to plot polynomials
    domain = LinRange(0,16,150)
    # plot degree 3 polynomial
    plt = plot(domain, p3.(domain); label="Degree 3 polynomial")
    # plot degree 6 polynomial
    plot!(domain, p6.(domain); label="Degree 6 polynomial")
    # plot degree 9 polynomial
    plot!(domain, p9.(domain); label="Degree 9 polynomial")
    # plot degree 12 polynomial
    plot!(domain, p12.(domain); label="Degree 12 polynomial")
    plot!(domain, jP13.(domain); label="Degree 13 polynomial")
    # plot discrete values
    scatter!(years, prices; label="Discrete Data", mc=:black, ms=2, ma=1)
    display(plt)
    readline()
    return
end

function plotEllipse(a,b,c,d,e)
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    θ = LinRange(0, 2π, 200)
    cosθ, sinθ = cos.(θ), sin.(θ)
    A = a.*cosθ.^2 .+ c.*cosθ.*sinθ .+ e.*sinθ.^2 
    B = b.*cosθ .+ d.* sinθ

    r =(-B .+ sqrt.(B.^2 .+ 4A)) ./ (2A)
    plt = plot(r.*cosθ, r.*sinθ; label="Best Fit", set_aspect=:equal)
    return plt
end

function ellipseFit()
    data = npzread("ellipse.npy")
    xVals = data[1:end, 1]
    yVals = data[1:end, 2]

    codomain = ones(size(xVals))
    A = [xVals.^2 xVals xVals .* yVals yVals yVals.^2]

    a,b,c,d,e = A\codomain

    plt = plotEllipse(a,b,c,d,e)
    scatter!(xVals, yVals; label="Descrete Points")
    title!("Ellipse Best Fit")
    display(plt)
    readline()
    return
end

test()

# ellipseFit()
# polynomialFit()
# lineFit()