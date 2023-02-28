# test_imageSegmentation.jl

include("imageSegmentation.jl")
using .ImageSegmentation


function test()
    A1 = [0 1 0 0 1 1; 1 0 1 0 1 0; 0 1 0 1 0 0; 0 0 1 0 1 1; 1 1 0 1 0 0; 1 0 0 1 0 0]
    L1 = [3 -1 0 0 -1 -1; -1 3 -1 0 -1 0; 0 -1 2 -1 0 0; 0 0 -1 3 -1 -1; -1 -1 0 -1 3 0; -1 0 0 -1 0 2]
    @assert L1 == laplacian(A1)

    A2 = [0 3 0 0 0 0; 3 0 0 0 0 0; 0 0 0 1 0 0; 0 0 1 0 2 0.5; 0 0 0 2 0 1; 0 0 0 0.5 1 0]
    L2 = [3 -3 0 0 0 0; -3 3 0 0 0 0; 0 0 1 -1 0 0; 0 0 -1 3.5 -2 -0.5; 0 0 0 -2 3 -1; 0 0 0 -0.5 -1 1.5]
    @assert L2 == laplacian(A2)

    @assert all((1,  1.5857864376269057) .≈ connectivity(A1))
    @assert all((2, 0) .== connectivity(A2))

    print("All tests passed")
    return
end

test()