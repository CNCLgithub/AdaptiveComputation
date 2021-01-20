export Object,
        Dot

# objects are things that dynamics models and generative processes
# work over (e.g. Dot)
abstract type Object end

@with_kw struct Dot <: Object
    pos::Vector{Float64} = zeros(3)
    vel::Vector{Float64} = zeros(3)
    acc::Vector{Float64} = zeros(3)
    probe::Bool = false
    radius::Float64 = 20.0
    width::Float64 = 40.0
    height::Float64 = 40.0
end

Dot(pos::Vector{Float64}, vel::Vector{Float64}) = Dot(pos = pos, vel = vel)
Dot(pos::Vector{Float64}, vel::Vector{Float64}, radius::Float64) = Dot(pos = pos, vel = vel,
                                                                       radius = radius,
                                                                       width = radius*2,
                                                                       height = radius*2)
