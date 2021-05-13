export Object,
        Dot,
        BDot

# objects are things that dynamics models and generative processes
# work over (e.g. Dot)
abstract type Thing end
abstract type Object <: Thing end
abstract type Ensemble <: Thing end

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


@with_kw struct Wall <: Object
    p1::Vector{Float64}
    p2::Vector{Float64}
    n::Vector{Float64} # wall normal
end


abstract type Polygon <: Object end


@with_kw mutable struct NGon <: Polygon
    pos::Vector{Float64}
    rot::Float64
    vel::Vector{Float64}
    ang_vel::Float64
    radius::Float64
    nv::Int64
end

@with_kw mutable struct UGon <: Polygon
    pos::Vector{Float64}
    vel::Vector{Float64}
end

get_pos(w::Wall) = (w.p2.+w.p1)/2
get_pos(d::Dot) = d.pos
get_pos(p::Polygon) = p.pos

#nv(p::Polygon)::Int64
nv(p::NGon) = p.nv
nv(p::UGon) = 1

radius(p::NGon) = p.radius
radius(p::UGon) = 0


struct UniformEnsemble
    rate::Float64
    pixel_prob::Float64
end

function UniformEnsemble(cg::CausalGraph)
    gm = get_gm(cg)
    graphics = get_graphics(cg)
    pixel_prob = (gm.dot_radius*pi^2)/prod(graphics.img_width)
    UniformEnsemble(gm.distractor_rate, pixel_prob)
end
