export InertiaGM, InertiaState

################################################################################
# Model definition
################################################################################

"""
Model that uses inertial change points to "explain" interactions
"""
@with_kw struct InertiaGM <: GenerativeModel

    # EPISTEMICS
    n_dots::Int64 = 2
    n_targets::Int64 = ceil(Int64, n_dots * 0.5)
    target_p::Float64 = n_targets / n_dots

    # DYNAMICS
    dot_mass::Float64 = 1.0
    dot_radius::Float64 = 20.0
    area_width::Float64 = 400.0
    area_height::Float64 = 400.0
    dimensions::Tuple{Float64, Float64} = (area_width, area_height)
    vel::Float64 = 10 # base vel
    bern::Float64 = 0.9
    k::Float64 = 100.0 # max von_misses kappa for angle
    w::Float64 = 2.5 # min standard deviation for magnitude noise
    # force parameters
    wall_rep_m::Float64 = 0.0
    wall_rep_a::Float64 = 0.02
    wall_rep_x0::Float64 = 0.0

    # GRAPHICS
    img_width::Int64 = 100
    img_height::Int64 = 100
    img_dims::Tuple{Int64, Int64} = (img_width,img_height)
    decay_rate::Float64 = 0.0
    min_mag::Float64 = 1E-4
    inner_f::Float64 = 1.0
    outer_f::Float64 = 1.0
    inner_p::Float64 = 0.95
    outer_p::Float64 = 0.3
    nlog_bernoulli::Float64 = -100
    bern_existence_prob::Float64 = -expm1(nlog_bernoulli)
end

function load(::Type{InertiaGM}, path::String; kwargs...)
    InertiaGM(;read_json(path)...,
              kwargs...)
end


struct InertiaState <: GMState
    walls::SVector{4, Wall}
    objects::Vector{Dot}
    ensemble::UniformEnsemble{Dot}
    es::RFSElements
    xs::AbstractArray
    pt::BitArray{3}
    pls::Vector{Float64}
end

function InertiaState(prev_st::InertiaState,
                      new_dots::Vector{Dot},
                      es::RFSElements{T},
                      xs::Vector{T}) where {T}
    (pls, pt) = GenRFS.associations(es, xs)
    setproperties(prev_st,
                  (objects = new_dots,
                   es = es,
                   xs = xs,
                   pt = pt,
                   pls = pls))
end

function InertiaState(gm::InertiaGM, dots::Vector{Dot})
    walls = init_walls(gm)
    n_ens = gm.n_dots - length(dots)
    InertiaState(walls, dots, UniformEnsemble(gm, n_ens),
                 RFSElements{Any}(undef, 0),
                 [],
                 falses(0,0,0),
                 Float64[])
end

function tracker_bounds(gm::InertiaState)
    @unpack area_width, area_height, dot_radius = gm
    xs = (-0.5*area_width + dot_radius, 0.5*area_width - dot_radius)
    ys = (-0.5*area_height + dot_radius, 0.5*area_height - dot_radius)
    (xs, ys, dot_radius)
end

function step(gm::InertiaGM,
              state::InertiaState,
              updates::Vector{KinematicsUpdate})

    # Dynamics (computing forces)
    # for each dot compute forces
    n_dots = length(updates)
    @unpack ensemble, walls, objects = state
    new_dots = Vector{Dot}(undef, n_dots)

    @inbounds for i in eachindex(objects)
        # force accumalator
        facc = MVector{2, Float64}(zeros(2))
        dot = objects[i]

        # interactions with walls
        for w in walls
            force!(facc, gm, w, dot)
        end

        # interactions with other dots
        for j = 1:n_dots
            i==j && continue
            force!(facc, gm, dot, objects[j])
        end

        # kinematics: resolve forces to pos vel
        (new_pos, new_vel) = update_kinematics(gm, dot, facc)

        # also do graphical update
        new_gstate = update_graphics(gm, dot, new_pos)
        new_dots[i] = update(dot, new_pos, new_vel, new_gstate)
    end

    return new_dots
end

normalvec(w::Wall, pos) = w.normal

"""Computes the force of A -> B"""
function force!(f::MVector{2, Float64}, dm::InertiaGM, ::Thing, ::Thing)
    return nothing
end

function force!(f::MVector{2, Float64}, dm::InertiaGM, w::Wall, d::Dot)
    @unpack pos = d
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    f .+= wall_rep_m * exp(-1 * (wall_rep_a * (d - wall_rep_x0))) * w.normal
    return nothing

end

# function force!(f::MVector{2, Float64}, dm::InertiaGM, a::Dot, b::Dot)
#     v = a.pos- b.pos
#     norm_v = norm(v)
#     absolute_force = dm.distance * exp(norm_v * dm.dot_repulsion)
#     f .+= absolute_force .* normalize(v)
#     return nothing
# end


function update_kinematics(::InertiaGM, ::Object, ::MVector{2, Float64})
    error("Not implemented")
end

function update_kinematics(gm::InertiaGM, d::Dot, f::MVector{2, Float64})
    # treating force directly as velocity; update velocity by x percentage; but f isn't normalized to be similar to v
    a = f/d.mass
    new_vel = d.vel + a
    new_pos = clamp.(d.pos + new_vel, -gm.area_height, gm.area_height)
    return new_pos, new_vel
end


function update_graphics(gm::InertiaGM, d::Dot, new_pos::SVector{2, Float64})

    @unpack area_width, area_height = gm
    @unpack img_width, img_height = gm

    # going from area dims to img dims
    x, y = translate_area_to_img(new_pos...,
                                 img_height, img_width,
                                 area_width, area_height)
    scaled_r = d.radius/area_width*img_width # assuming square
    gstate = exp_dot_mask(x, y, scaled_r, img_width, img_height, gm)

    # Mario: trying to deal with segf when dropping
    decayed = deepcopy(d.gstate)
    rmul!(decayed, gm.decay_rate)
    droptol!(decayed, gm.min_mag)

    # overlay new render onto memory

    #without max, tail gets lost; . means broadcast element-wise
    max.(gstate, decayed)
    #map!(max, gstate, gstate, decayed)
    return gstate
end


function predict(gm::InertiaGM, st::InertiaState)::RFSElements{BitMatrix}
    @unpack objects, ensemble = st
    n = length(st.objects)
    es = RFSElements{BitMatrix}(undef, n + 1)
    @inbounds for i in 1:n
        obj = st.objects[i]
        es[i] = BernoulliElement{BitMatrix}(nlog_bernoulli,
                                            mask,
                                            (obj.gstate,))
    end
    es[n + 1] = PoissonElement{BitMatrix}(ensemble.rate,
                                          mask,
                                          (ensemble.gstate,))
    return es
end

# function render(gm::RepulsionGM, st::RepulsionState)

#     gstate = zeros(gm.img_dims)
#     @inbounds for j = 1:gm.n_dots
#         #dot = ?
#         dot = current_state.objects[j]
#         gstate .+= dot.gstate
#     end
#     rmul!(gstate, 1.0 / gm.n_dots)
#     Gray.(gstate)
# end

include("helpers.jl")
include("gen.jl")
