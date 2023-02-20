module StandardLibary

include("./calculator.jl")
include("./box.jl")

using .Box
using .Calculator
using Statistics
using Combinatorics

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
    B = collect(A)
    powerSet = collect(powerset(B))
    C = Set()
    for set in powerSet
        push!(C, Set(set))
    end
    return C
end

function main(args)

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

function shutTheBox(name, timeLimit)

    remainingNumbers = [1:9;]
    len = size(remainingNumbers, 1)
    startTime = time()
    endTime = 0

    while length(remainingNumbers) > 0 && ((endTime-startTime) <= timeLimit)
        println("Numbers left: ", remainingNumbers)
        if Base.sum(remainingNumbes) < 6
            roll = rand(1:6)
        else
            roll = rand(1:12)
        end
        println("roll: ", roll)

        if !(isValid(roll, remainingNumbers))
            println("Game Over!")
            endTime = time()
            break
        end

        if len == length(remainingNumbers)
            println("Seconds left: ", timeLimit)
        else
            println("Seconds left: ", round(int(timeLimit) - endTime + startTime; digits=2))
        end
        playerNumbers = Vecotr{Float64}()

        while length(playerNumbers) == 0
            println("Numbers to eliminate: ")

            playerInput = readline()
            playerNumbers = parseInput(playerInput, remainingNumbers)

            if Base.sum(playerNumbers) != roll
                endTime = time()

                if !((endTime - startTime) <= int(timeLimit)) 
                    break
                end
                println("Invalid inputs")
                println("Seconds left: ", round(int(timeLimit) - endTime + startTime; digits=2))
                playerNumbers = []
            end
        end
        for num in enumerate(playerNumbers)
            filter!(x -> x != num, remainingNumbers)
        end
        endTime = time()
        println("\n")
    end
    println("Score for player ", name, ": ", Base.sum(remainingNumbers))
    println("Time played: ", round(endTime - startTime; digits=2))

    if Base.sum(remainingNumbers) == 0 && ((endTime - startTime) <= int(timeLimit))
        println("Congratulations!! You shut the box!")
    else
        println("Better luck next time! >:(")
    end

    return
end


if length(ARGS) == 2

    name = ARGS[2]
    time = ARGS[end]

    shutTheBox(name, time)
else
    main(ARGS)
end

end
