module Box

using Combinatorics
export isValid, parseInput

function isValid(roll, remaining)

    if !(roll in 1:12)
        return false
    else
        for i=1:length(remaining)
            if any([sum(combo) == roll for combo in combinations(remaining, i)])
                return true
            end
        end
    end
end

function parseInput(playerInput, remaining)
    try
        choices = [int(i) for i in split(playerInput, " ")]
        if length(Set(choices)) != length(choices)
            throw(choices)
        end
        if any([!(number in remaining) for number in chioces])
            throw(choices)
        end
        return choices
    catch
        @warn "some user inputs are incorrect returning empty choice array"
        return []
    end
end

end