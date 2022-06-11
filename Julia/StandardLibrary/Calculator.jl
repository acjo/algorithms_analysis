module Calculator

using Base ; sqrt

export sqrt, sum

function sum(x, y)
    return x + y
end

function prod(x, y)
    return x*y
end


end