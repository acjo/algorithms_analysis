# test_lstsqEigs.jl

include("lstsqEigs.jl")
using .LeastSquaresEigenvalues 
using LinearAlgebra
using Random
using NPZ
using Plots

function test()

    d = MersenneTwister(0)
    A = rand(d, Float64, (6, 4))
    b = rand(d, Float64, (6,))

    x = leastSquareSol(A, b)

    y = A\b

    @assert x ≈ y
    
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


test()

lineFit()