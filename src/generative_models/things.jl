export Thing, Object, Dot, Wall, UniformEnsemble

using DataStructures

abstract type Thing end

abstract type Object <: Thing end

"""
The probability that the object is a target. Zero default
"""
function target(::Object)
    return 0.0
end

function get_pos(::Object)
    error("Not implemented")
end

function get_vel(::Object)
    error("Not implemented")
end

@with_kw struct Dot <: Object
    # Dynamics
    radius::Float64
    mass::Float64

    # Kinematics
    vel::SVector{2, Float64}
    tail::CircularBuffer{SVector{2, Float64}}

    # Graphics
    gstate::Vector{GaussianComponent{2}}

    # Misc
    target::Float64
end

function _Dot(radius::Float64, mass::Float64, pos::SVector{2, Float64},
             vel::SVector{2, Float64}, nt::Int, target::Bool)
    tail = CircularBuffer{SVector{2, Float64}}(nt)
    pushfirst!(tail, pos)
    g = GaussianComponent{SVector{2, Float64}}[]
    Dot(radius, mass, vel, tail, g, target)
end

get_pos(d::Dot) = first(d.tail)
get_vel(d::Dot) = d.vel
target(d::Dot) = d.target

struct Wall <: Object
    # Dynamics
    d::Float64 # the distance from the center
    normal::SVector{2, Float64} # normal vector
    nd::SVector{2, Float64}
    function Wall(d::Float64, normal::SVector{2, Float64})
        new(d, normal, d * normal)
    end
    # Kinematics <none>
    # Graphcis <none>
end

const WALL_ANGLES = [0, pi/2, pi, 3/2 * pi]

function init_walls(width::Float64)
   ws = Vector{Wall}(undef, 4)
   d = [ width * .5,
         width * .5,
        -width * .5,
        -width * .5]
   @inbounds for (i, theta) in enumerate(WALL_ANGLES)
        normal = SVector{2, Float64}([cos(theta), sin(theta)])
        ws[i] = Wall(d[i], normal)
    end
    return SVector{4, Wall}(ws)
end

abstract type Ensemble <: Object end

@with_kw struct UniformEnsemble <: Ensemble
    rate::Float64
    pixel_prob::Float64
    targets::Int64 = 0
end

target(u::UniformEnsemble) = u.rate === 0. ? 0. : u.targets / u.rate

const ZERO_POS = SVector{2, Float64}(zeros(2))
get_pos(e::UniformEnsemble) = ZERO_POS
