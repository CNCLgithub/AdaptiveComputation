export Object,
        Dot,
        BDot

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


@with_kw struct Pylon <: Object
    pos::Vector{Float64} = zeros(3)
    radius::Float64 = 40.0
    strength::Float64 = 10.0
end

Dot(pos::Vector{Float64}, vel::Vector{Float64}) = Dot(pos = pos, vel = vel)
Dot(pos::Vector{Float64}, vel::Vector{Float64}, radius::Float64) = Dot(pos = pos, vel = vel,
                                                                       radius = radius,
                                                                       width = radius*2,
                                                                       height = radius*2)

# dot with bearing
mutable struct BDot <: Object
    pos::Vector{Float64}
    bearing::Float64
    vel::Float64
end


@with_kw mutable struct Polygon <: Object
    pos::Vector{Float64}
    vel::Vector{Float64}
    rot::Float64
    ang_vel::Float64
    radius::Float64
    dots::Vector{Dot}
end
