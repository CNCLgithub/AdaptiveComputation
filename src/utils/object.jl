export Object,
        Dot,
        BDot

using LightGraphs
using MetaGraphs
#const CausalGraph = MetaGraphs.MetaDiGraph{Int64, Vector{Float64}}
const CausalGraph = MetaGraphs.MetaDiGraph{Int64, Float64}

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


# assuming first N vertices are walls
walls(cg::CausalGraph) = get_prop(cg, :walls)

function force(cg::CausalGraph, v::Int64)
    fs = @>> v begin
        inneighbors(cg)
        map(i -> Edge(i, v))
        Base.filter(e -> has_prop(cg, e, :force))
        map(e -> get_prop(cg, e, :force))
    end

    return isempty(fs) ? zeros(2) : sum(fs)
end

LightGraphs.vertices(cg::CausalGraph, v::Int64) = @>> v begin
    outneighbors(cg)
    collect(Int64)
    Base.filter(i -> has_prop(cg, Edge(v, i), :parent))
end


parent(cg::CausalGraph, v::Int64) = @>> v begin
    inneighbors(cg)
    Base.filter(i -> has_prop(cg, Edge(i, v), :parent))
    first
    #(i -> get_prop(cg, i, :object))
end

get_objects(cg::CausalGraph, type::Type) = @>> cg begin
    vertices
    map(v -> get_prop(cg, v, :object))
    Base.filter(v -> v isa type)
end

function get_object_verts(cg::CausalGraph, type::Type)
    filter_vertices(cg, (g, v) -> get_prop(g, v, :object) isa type)
end
