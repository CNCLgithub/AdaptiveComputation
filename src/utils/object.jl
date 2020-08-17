export Object,
        Dot,
        BDot

# objects are things that dynamics models and generative processes
# work over (e.g. Dot)
abstract type Object end

mutable struct Dot <: Object
    pos::Vector{Float64}
    vel::Vector{Float64}
end


# dot with bearing
mutable struct BDot <: Object
    pos::Vector{Float64}
    bearing::Float64
    vel::Float64
end
