# imageSegmentation.jl

module ImageSegmentation

using Random
using Plots
using LinearAlgebra

export laplacian, connectivity, ImageSegmenter

function laplacian(A::Matrix{<:Real})
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) Matrix{itn}): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    # sum across rows
    L1 = vec(sum(A; dims=1))
    # create degree matrix
    D = diagm(L1)
    # return Laplacian
    return D - A
end

function connectivity(A::Matrix{<:Real}; tol::Float64=1e-8)
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) Matrix): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """

    # get the laplacian
    L = laplacian(A)
    # compute the spectral decomposition
    F = eigen(L)
    # extract the eigen values (convert them to real)
    eigenValues = real.(F.values)

    mask = eigenValues .< tol
    numConnected = sum(mask)
    eigenValues[mask] .= 0

    # if the number of connected components is greater than or equal to 2 then the
    # connectivity is obviously equal to zero.
    if numConnected >= 2
        connectivity = 0
    # otherwise get the second smalles eigenValue which will (which will have to be larger than zero)
    else
        # get the index sorting (smallest => largest)
        argsort = sortperm(eigenValues)
        # extract the index 
        ii = argsort[2]
        # extract connectivity
        connectivity = eigenValues[ii]
    end

    return numConnected, connectivity
end

function getNeighbors(index, radius, height, width) 
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel
    # CONSIDER: incremented by 1 because floor could round to 0 and index % width could be zero
    # which are not valid index in Julia arrays
    row, col = Int(floor(index / width)), index % width
    row += 1
    col += 1

    # get a grid of possible candidates that are close to the central pixel
    r = Int(radius)

    x = range(max(col - r, 1), min(col + r + 1, width); step=1) |> collect
    y = range(max(col - r, 1), min(col + r + 1, height); step=1) |> collect

    # create our meshgrid
    X = x' .* ones(size(x,1))
    Y = ones(size(y,1))' .* y

    # determine which candidates are in the radius of the central pixel
    R = sqrt.((X - col).^2 .+ (Y-row).^2)
    mask = R .< radius

    return Int.(X[mask] + Y[mask] .* width), R[mask]
end


mutable struct ImageSegmenter
    fileName::Union{Nothing, String}
    scaled::Union{Nothing,Matrix{RGBA{Float32}}, Matrix{RGBA{Float64}}}
    rgb::Union{Nothing, Bool}
    brightness::Union{Nothing, Matrix{}, Vector{}}

    function ImageSegmenter(;fileName=nothing, scaled=nothing,rgb=nothing, brightness=nothing)
        """
        Constructor if nothing is provided in the instantiation. I.e. ImageSegmenter()
        """
        return new(nothing, nothing, nothing,nothing)
    end
    function ImageSegmenter(fileName; scaled=nothing,rgb=nothing, brightness=nothing)
        """
        Constructor if only the file name is provided in the instantiation. I.e. ImageSegmenter(fileName)
        """

        # load in image
        scaled = load(fileName)
        # scale the image
        scaled /= 255


        return new(fileName, scaled, nothing,nothing)
    end
end


    
end