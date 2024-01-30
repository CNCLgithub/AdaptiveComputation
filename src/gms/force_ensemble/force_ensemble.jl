export ForceEnsemble

"""
Model that computes force between objects
"""
@with_kw struct ForceEnsemble <: GenerativeModel

    # EPISTEMICS
    n_dots::Int64 = 2
    n_targets::Int64 = ceil(Int64, n_dots * 0.5)
    target_p::Float64 = n_targets / n_dots

    # DYNAMICS
    dot_radius::Float64 = 20.0
    area_width::Float64 = 800.0
    area_height::Float64 = 800.0
    dimensions::Tuple{Float64, Float64} = (area_width, area_height)
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance_factor::Float64 = 10.0
    max_distance::Float64 = 100.0
    vel::Float64 = 10.0 # base velocity
    rep_inertia::Float64 = 0.9
    force_sd::Float64 = 1.0
    ens_repulsion::Float64 = 1.0
    ens_force_sd::Float64 = 0.9
    ens_sd::Float64 = 0.9

    # GRAPHICS
    img_width::Int64 = 100
    img_height::Int64 = 100
    img_dims::Tuple{Int64, Int64} = (img_width,img_height)
    min_mag::Float64 = 1E-4
    inner_f::Float64 = 1.0
    outer_f::Float64 = 1.0
    inner_p::Float64 = 0.95
    outer_p::Float64 = 0.3
    k_tail::Int64 = 4 # number of point in tail
    tail_sample_rate::Int64 = 2
end

struct GaussianEnsemble <: Ensemble
    mu::SVector{2, Float64}
    sigma::Float64
    rate::Float64
end
target(::GaussianEnsemble) = 0.0
get_pos(e::GaussianEnsemble) = e.mu
get_vel(e::GaussianEnsemble) = ZERO_VEC2

struct ForceEState <: GMState{ForceEnsemble}
    walls::SVector{4, Wall}
    objects::Vector{Dot}
    ensemble::GaussianEnsemble
end

get_objects(st::ForceEState) = st.objects

function ForceEState(gm::ForceEnsemble,
                     objects::AbstractVector{<:Dot},
                     ensemble::GaussianEnsemble)
    walls = init_walls(gm.area_width)
    ForceEState(walls, objects, ensemble)
end


function ForceEState(gm::ForceEnsemble,
                     objects::AbstractVector{<:Dot})
    # bumping to ensure non-zero rate
    rate = Float64(gm.n_dots - length(objects)) + 0.01
    ensemble = GaussianEnsemble(ZERO_VEC2, gm.area_width, rate)
    ForceEState(gm, objects, ensemble)
end

function tracker_bounds(gm::ForceEnsemble)
    @unpack area_width, area_height, dot_radius = gm
    xs = (-0.5*area_width + dot_radius, 0.5*area_width - dot_radius)
    ys = (-0.5*area_height + dot_radius, 0.5*area_height - dot_radius)
    (xs, ys)
end

function Dot(gm::ForceEnsemble,
             pos::SVector{2, Float64},
             vel::SVector{2, Float64},
             target::Bool)
    t_dot = _Dot(gm.dot_radius, 1.0, pos, vel,
                gm.k_tail, target)
    update_graphics(gm, t_dot)
end

function step(gm::ForceEnsemble,
              state::ForceEState,
              forces::AbstractVector{<:SVector{2, Float64}},
              new_ensemble::GaussianEnsemble)

    objects = state.objects
    n_dots = length(objects)
    n_dots == length(forces) ||
        error("Length of `forces` missmatch with objects")

    new_dots = Vector{Dot}(undef, n_dots)
    ensemble_acc = MVector{2, Float64}(0., 0.)
    @inbounds for i in eachindex(objects)
        dot = objects[i]
        # force accumalator
        facc = MVector{2, Float64}(forces[i])
        # interactions with walls
        for w in state.walls
            force!(facc, gm, w, dot)
        end
        for j in eachindex(objects)
            i === j && continue
            force!(facc, gm, objects[j], dot)
        end
        # ensemble -> dot
        force!(facc, gm, new_ensemble, dot)
        # kinematics: resolve forces to pos vel
        ku = update_kinematics(gm, dot, facc)
        new_dots[i] = update_graphics(gm, sync_update(dot, ku))
    end

    setproperties(state; objects = new_dots,
                  ensemble = new_ensemble)
end

function step(gm::ForceEnsemble,
              state::ForceEState)
    obj_forces = Fill(ZERO_VEC2, length(state.objects))
    step(gm, state, obj_forces, state.ensemble)
end

function force!(f::MVector{2, Float64}, gm::ForceEnsemble, w::Wall, d::Dot)
    pos = get_pos(d)
    @unpack wall_repulsion, max_distance, distance_factor = gm
    v = LinearAlgebra.norm(w.normal .* pos + w.nd)
    v > max_distance && return nothing
    # mag = wall_repulsion * exp(-v/distance_factor)
    mag = exp(-((v - wall_repulsion)/distance_factor))
    f .+= mag * w.normal
    return nothing
end

function force!(f::MVector{2, Float64}, gm::ForceEnsemble, x::Dot, d::Dot)
    @unpack dot_repulsion, max_distance, distance_factor = gm
    v = get_pos(d) - get_pos(x)
    nv = norm(v)
    nv > max_distance && return nothing
    # mag = dot_repulsion * exp(-nv/distance_factor)
    mag = exp(-((nv - dot_repulsion)/distance_factor))
    delta_f = mag * (v./nv)
    f .+= delta_f
    return nothing
end

function force!(f::MVector{2, Float64}, gm::ForceEnsemble,
                x::GaussianEnsemble, d::Dot)
    @unpack ens_repulsion, max_distance, distance_factor = gm
    v = get_pos(d) - get_pos(x)
    nv = norm(v)
    nv > max_distance && return nothing
    # scaled = nv * (x.sigma) / x.rate
    # mag = exp(-((scaled - ens_repulsion)/distance_factor))
    # delta_f = mag * (v./nv)
    # f .+= delta_f
    return nothing
end


# function force!(f::MVector{2, Float64}, gm::ForceEnsemble,
#                 d::Dot,
#                 x::GaussianEnsemble)
#     force!(f, gm, x, d)
#     return nothing
# end

# function force!(f::MVector{2, Float64}, gm::ForceEnsemble,
#                 w::Wall, e::GaussianEnsemble)
#     pos = get_pos(e)
#     @unpack wall_repulsion, max_distance, distance_factor = gm
#     v = LinearAlgebra.norm(w.normal .* pos + w.nd)
#     v > max_distance && return nothing
#     scaled = v * e.sigma / x.rate
#     # mag = wall_repulsion * exp(-v/distance_factor)
#     mag = exp(-((scaled - wall_repulsion)/distance_factor))
#     f .+= mag * w.normal
#     return nothing
# end

function update_kinematics(gm::ForceEnsemble, d::Dot, f::MVector{2, Float64})
    @unpack rep_inertia, vel, area_height = gm
    nf = max(norm(f), 0.01)
    f_adj = f .* (min(nf, rep_inertia) / nf)
    v = get_vel(d) + f_adj
    nv = max(norm(v), 0.01)
    new_vel = v .* (clamp(nv, gm.vel, 3 * gm.vel) / nv)
    prev_pos = get_pos(d)
    new_pos = clamp.(prev_pos + new_vel,
                     -area_height * 0.5 + d.radius,
                     area_height * 0.5  - d.radius)
    KinematicsUpdate(new_pos, new_vel)
end

function update_ensemble(gm::ForceEnsemble, e::GaussianEnsemble,
                         dx::Float64, dy::Float64, ds::Float64)

    d_mu = SVector{2, Float64}(10.0 * dx, 10.0 * dy)
    new_mu = e.mu + d_mu
    new_sigma = clamp(e.sigma + ds, 0.0, 0.5 * gm.area_height)
    GaussianEnsemble(new_mu, new_sigma, e.rate)
end

# function update_kinematics(gm::ForceEnsemble, e::GaussianEnsemble,
#                            f::MVector{2, Float64})
#     @unpack rep_inertia, vel, area_height = gm
#     nf = max(norm(f), 0.01)
#     # assumes 0 vel
#     v = f .* (min(nf, rep_inertia) / nf)
#     nv = max(norm(v), 0.01)
#     new_vel = v .* (clamp(nv, gm.vel, 3 * gm.vel) / nv)
#     prev_pos = get_pos(e)
#     new_pos = clamp.(prev_pos + new_vel,
#                      -area_height * 0.5 + e.sigma,
#                      area_height * 0.5  - e.sigma)
#     KinematicsUpdate(new_pos, new_vel)
# end

function update_graphics(gm::ForceEnsemble, d::Dot)
    @unpack inner_f, outer_f, tail_sample_rate = gm
    nt = length(d.tail)
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


function rf_elements(gm::ForceEnsemble, objects::AbstractVector{Dot})
    n = length(objects)
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n)
    # the trackers
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp, (obj.gstate,))
    end
    return es
end

function rf_elements(gm::ForceEnsemble, objects::AbstractVector{Dot},
                     e::GaussianEnsemble)
    n = length(objects)
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n + 1)
    # the trackers
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp, (obj.gstate,))
    end
    # ensemble
    nt = length(objects[1].gstate)
    cov = SMatrix{2,2,Float64}(spdiagm(Fill(e.sigma, 2)))
    e_gpp = Fill(GaussianComponent{2}(1.0, e.mu, cov), nt)
    es[n + 1] = PoissonElement{GaussObs{2}}(e.rate, gpp, (e_gpp,))
    return es
end


function predict(gm::ForceEnsemble, st::ForceEState)
    rf_elements(gm, st.objects, st.ensemble)
end

function observe(gm::ForceEnsemble,
                 objects::AbstractVector{Dot})
    es = rf_elements(gm, objects)
    (es, gpp_mrfs(es, 50, 1.0))
end

include("gen.jl")

gen_fn(::ForceEnsemble) = gm_force_ensemble
const ForceEnsembleIr = Gen.get_ir(gm_force_ensemble)
const ForceEnsembleTrace = Gen.get_trace_type(gm_force_ensemble)


################################################################################
# Misc
################################################################################

function objects_from_positions(gm::ForceEnsemble, positions, targets)
    nx = length(positions)
    dots = Vector{Dot}(undef, nx)
    for i = 1:nx
        xy = Float64.(positions[i][1:2])
        dots[i] = Dot(gm,
                      SVector{2}(xy),
                      SVector{2, Float64}(zeros(2)),
                      targets[i])
    end
    return dots
end

function state_from_positions(gm::ForceEnsemble, positions, targets)
    targets = collect(Bool, targets) # this is sometimes Int
    nt = length(positions)
    states = Vector{ForceEState}(undef, nt)
    for t = 1:nt
        if t == 1
            dots = objects_from_positions(gm, positions[t], targets)
            states[t] = ForceEState(gm, dots)
            continue
        end

        prev_state = states[t-1]
        @unpack objects = prev_state
        ni = length(objects)
        new_dots = Vector{Dot}(undef, ni)
        for i = 1:ni
            old_pos = Float64.(positions[t-1][i][1:2])
            new_pos = SVector{2}(Float64.(positions[t][i][1:2]))
            new_vel = SVector{2}(new_pos .- old_pos)
            dot = sync_update(objects[i], KinematicsUpdate(new_pos, new_vel))
            new_dots[i] = update_graphics(gm, dot)
        end
        states[t] = ForceEState(gm, new_dots)
    end
    return states
end

function extract_rfs_subtrace(trace::ForceEnsembleTrace, t::Int64)
    # StaticIR names and nodes
    # gen_fn = get_gen_fn(trace)
    outer_ir = Gen.get_ir(gm_force_ensemble)
    kernel_node = outer_ir.call_nodes[2] # (:kernel)
    kernel_field = Gen.get_subtrace_fieldname(kernel_node)
    # subtrace for each time step
    vector_trace = getproperty(trace, kernel_field)
    sub_trace = vector_trace.subtraces[t]
    # StaticIR for `force_kernel`
    inner_ir = Gen.get_ir(fe_kernel)
    xs_node = inner_ir.call_nodes[3] # (:masks)
    xs_field = Gen.get_subtrace_fieldname(xs_node)
    # `RFSTrace` for :masks
    getproperty(sub_trace, xs_field)
end

