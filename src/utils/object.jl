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


struct Wall <: Object
    x::Float64
    y::Float64
end



abstract type Polygon <: Object end

nv(p::Polygon)::Int64
nv(p::NGon) = p.nv
nv(p::UGon) = 0

radius(p::NGon) = p.radius
radius(p::UGon) = 0


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
    rot::Float64
    vel::Vector{Float64}
    ang_vel::Float64
    radius::Float64
    nv::Int64
end

const CausalGraph = MetaGraphs.MetaDiGraph{Int64, Vector{Float64}}

# assuming first N vertices are walls
walls(cg::CausalGraph) = get_prop(cg, :walls)

force(cg::CausalGraph, v::Int64) = @>> v begin
    inneighbors(cg)
    map(i -> Edge(i, v))
    Base.filter(e -> has_prop(cg, e, :force))
    map(get_prop(cg, e, :force))
    sum
end

vertices(cg::CausalGraph, v::Int64) = @>> v begin
    outneighbors(cg)
    collect(Int64)
    Base.filter(i -> has_prop(cg, Edge(v, i), :parent))
end


parent(cg::CausalGraph, v::Int64) = @>> v begin
    inneighbors(cg)
    Base.filter(i -> has_prop(cg, Edge(i, v), :parent))
    first
    (i -> get_prop(cg, i, :object))
end
