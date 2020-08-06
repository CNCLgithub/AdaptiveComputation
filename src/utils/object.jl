export Object,
        Dot

# objects are things that dynamics models and generative processes
# work over (e.g. Dot)
abstract type Object end

mutable struct Dot <: Object
    pos::Vector{Float64}
    vel::Vector{Float64}
    # radius::Float64
end
