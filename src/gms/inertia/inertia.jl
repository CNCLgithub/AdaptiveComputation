export InertiaGM, InertiaState, load

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
    decay_rate::Float64 = 0.1
    min_mag::Float64 = 1E-4
    inner_f::Float64 = 1.0
    outer_f::Float64 = 1.0
    inner_p::Float64 = 0.95
    outer_p::Float64 = 0.3
    k_tail::Int64 = 4 # number of point in tail
    tail_sample_rate::Int64 = 2
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
    ensemble::UniformEnsemble
    es::RFSElements
    xs::AbstractArray
    pt::BitArray{3}
    pls::Vector{Float64}
end

struct KinematicsUpdate
    p::SVector{2, Float64}
    v::SVector{2, Float64}
end

# function step(state::InertiaState, updates)
#     step(state.gm, state, updates)
# end

function step(gm::InertiaGM,
              state::InertiaState,
              updates)
              # updates::AbstractVector{KinematicsUpdate})

    # Dynamics (computing forces)
    # for each dot compute forces
    n_dots = length(updates)
    @unpack ensemble, walls, objects = state
    new_dots = Vector{Dot}(undef, n_dots)

    @inbounds for i in eachindex(objects)
        # force accumalator
        facc = MVector{2, Float64}(zeros(2))
        # applies the motion kernel from gen
        dot = overwrite_update(objects[i], updates[i])
        # interactions with walls
        for w in walls
            force!(facc, gm, w, dot)
        end
        # kinematics: resolve forces to pos vel
        ku = update_kinematics(gm, dot, facc)
        dot = sync_update(dot, ku)
        # also do graphical update
        new_dots[i] = update_graphics(gm, dot)
    end
    return new_dots
end

"""Computes the force of A -> B"""
function force!(f::MVector{2, Float64}, dm::InertiaGM, ::Thing, ::Thing)
    return nothing
end

function force!(f::MVector{2, Float64}, dm::InertiaGM, w::Wall, d::Dot)
    pos = get_pos(d)
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    f .+= wall_rep_m * exp(-1 * (wall_rep_a * (n - wall_rep_x0))) * w.normal
    return nothing

end

# function force!(f::MVector{2, Float64}, dm::InertiaGM, a::Dot, b::Dot)
#     v = get_pos(a)- get_pos(b)
#     norm_v = norm(v)
#     absolute_force = dm.distance * exp(norm_v * dm.dot_repulsion)
#     f .+= absolute_force .* normalize(v)
#     return nothing
# end

function overwrite_update(d::Dot, ku::KinematicsUpdate)
    # replace the current head with ku
    cb = deepcopy(d.tail)
    cb[1] = ku.p
    setproperties(d, (vel = ku.v, tail = cb))
end

"""
    update_kinematics(::GM, ::Object, ::MVector{2, Float64})

resolve force on object, returning kinematics update
"""
function update_kinematics end

function update_kinematics(gm::InertiaGM, d::Dot, f::MVector{2, Float64})
    # treating force directly as velocity;
    # update velocity by x percentage;
    # but f isn't normalized to be similar to v
    a = f/d.mass
    new_vel = d.vel + a
    new_pos = clamp.(get_pos(d) + new_vel,
                     -gm.area_height * 0.5 + d.radius,
                     gm.area_height * 0.5  - d.radius)
    KinematicsUpdate(new_pos, new_vel)
end


function update_graphics(gm::InertiaGM, d::Dot)

    nt = length(d.tail)

    @unpack area_width, decay_rate = gm
    @unpack inner_f, outer_f, tail_sample_rate = gm
    r = d.radius
    base_sd = r * inner_f
    nk = ceil(Int64, nt / tail_sample_rate)
    gpoints = Vector{GaussianComponent{2}}(undef, nk)
    # linearly increase sd
    step_sd = (outer_f - inner_f) * r / nt
    c::Int64 = 1
    i::Int64 = 1
    @inbounds while c <= nt
        c_next = min(nt, c + tail_sample_rate - 1)
        pos = mean(d.tail[c:c_next])
        sd = (i-1) * step_sd + base_sd
        cov = SMatrix{2,2}(spdiagm([sd, sd]))
        gpoints[i] = GaussianComponent{2}(1.0, pos, cov)
        c = c_next + 1
        i += 1
    end
    setproperties(d, (gstate = gpoints))
end

function predict(gm::InertiaGM,
                 t::Int,
                 st::InertiaState,
                 objects::AbstractVector{Dot})
    n = length(objects)
    es = RFSElements{GaussObs{2}}(undef, n + 1)
    # the trackers
    @unpack area_width, k_tail, tail_sample_rate = gm
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp,
                                       (obj.gstate,))
    end
    # the ensemble
    tback = t < k_tail ? (t + 1) : k_tail
    nt = ceil(Int64, tback / tail_sample_rate)
    w = -log(nt) # REVIEW: no longer used in `GaussianComponent`
    @unpack rate = (st.ensemble)
    mu = @SVector zeros(2)
    cov = SMatrix{2,2}(spdiagm([50*area_width, 50*area_width]))
    uniform_gpp = Fill(GaussianComponent{2}(w, mu, cov), nt)
    es[n + 1] = PoissonElement{GaussObs{2}}(rate, gpp, (uniform_gpp,))
    return es
end

function observe(gm::InertiaGM,
                 objects::AbstractVector{Dot})
    n = length(objects)
    es = RFSElements{GaussObs{2}}(undef, n)
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp, (obj.gstate,))
    end
    (es, gpp_mrfs(es, 50, 1.0))
end

include("helpers.jl")
include("gen.jl")
