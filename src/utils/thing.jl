export Object,
        Dot

abstract type Thing end
abstract type Object <: Thing end

@with_kw struct Dot <: Object
    pos::Vector{Float64} = zeros(3)
    vel::Vector{Float64} = zeros(3)
    acc::Vector{Float64} = zeros(3)
    probe::Bool = false
    radius::Float64 = 20.0
    width::Float64 = 40.0
    height::Float64 = 40.0
end

# Dot(pos::Vector{Float64}, vel::Vector{Float64}) = Dot(pos = pos, vel = vel)
# Dot(pos::Vector{Float64}, vel::Vector{Float64}, radius::Float64) = Dot(pos = pos, vel = vel,
                                                                       # radius = radius,
                                                                       # width = radius*2,
                                                                       # height = radius*2)

@with_kw struct Wall <: Object
    p1::Vector{Float64}
    p2::Vector{Float64}
    n::Vector{Float64} # wall normal pointing inwards
end

function init_walls(area_width::Float64, area_height::Float64)
    ws = Vector{Wall}(undef, 4)
    ps = Iterators.product([-area_width/2, +area_width/2], [-area_height/2, +area_height/2])
    ps = @>> ps map(x -> [x[1], x[2]]) vec
    combs = [[1,2], [1,3], [2,4], [3,4]]
    for i=1:4
        p1 = ps[combs[i][1]]
        p2 = ps[combs[i][2]]
        ws[i] = Wall(p1, p2, _get_wall_normal(p1, p2, area_width, area_height))
    end
    return ws
end

function _contains(p::Vector{Float64}, area_width::Real, area_height::Real)::Bool
    xmin, xmax = (-area_width/2, area_width/2)
    ymin, ymax = (-area_height/2, area_height/2)

    xcheck = p[1] >= xmin && p[1] <= xmax
    ycheck = p[2] >= ymin && p[2] <= ymax

    return xcheck && ycheck
end
function _get_wall_normal(p1::Vector{Float64}, p2::Vector{Float64},
                          area_width::Real, area_height::Real)::Vector{Float64}
    wall_vec = p2 .- p1
    x = wall_vec[2]/norm(wall_vec)
    y = -wall_vec[1]/norm(wall_vec)
    n = [x,y]

    return _contains(n, area_width, area_height) ? n : -n
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

nv(p::NGon) = p.nv
nv(p::UGon) = 1

radius(p::NGon) = p.radius
radius(p::UGon) = 0


abstract type Ensemble <: Thing end

struct UniformEnsemble <: Ensemble
    rate::Float64
    pixel_prob::Float64
end

function UniformEnsemble(cg)
    gm = get_gm(cg)
    graphics = get_graphics(cg)
    # just the threshold from receptive_fields
    pixel_prob = @>> graphics.receptive_fields first (x -> x.threshold)
    UniformEnsemble(gm.distractor_rate, pixel_prob)
end

get_pos(e::UniformEnsemble) = [0,0,-Inf]
