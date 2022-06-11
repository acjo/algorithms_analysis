module StandardLibary
include("./Calculator.jl")
using Statistics


function prob1(L)

    return minimum(L), maximum(L), Statistics.mean(L)

end

function prob2()

    x = 3

    if ismutable(x)
        println("integers are mutable")
    else
        println("integers are immutable")
    end

    x = "hello"

    if ismutable(x)
        println("strings are mutable")
    else
        println("strings are immutable")
    end


    x = [1, 2, 3]

    if ismutable(x)
        println("arrays are mutable")
    else
        println("Arrays are immutable")
    end

    x = (1, 2, 3)

    if ismutable(x)
        println("tuples are mutable")
    else
        println("tuples are immutable")

    end

    x = Set(x)

    if ismutable(x)
        println("sets are mutable")
    else
        println("sets are immutable")

    end

end

function prob3(a, b)
    return Calculator.sqrt(Calculator.sum(Calculator.prod(a, a), Calculator.prod(b, b)))
end

function prob4(A)

    power_set = Vector{Set}(nothing, 2^length(A))
    for i in eachIndex(A)

        if i == 1
            power_set[i] = Set()

        end


    end

    return power_set

end


function test(args)

    if length(args) == 0
        return

    elseif args[1] == "1"
        println(prob1([1, 2, 3, 4]))

    elseif args[1] == "2"
        prob2()

    elseif args[1] == "3"
        println(prob3(3, 4))

    elseif args[1] == "4"
        A = ['a', 'b', 'c']
        println(prob4(A))

    end

end


println(typeof(ARGS))
test(ARGS)

end
