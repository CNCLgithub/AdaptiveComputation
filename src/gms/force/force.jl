export ForceGM

"""
Model that computes force between objects
"""
@with_kw struct ForceGM <: GenerativeModel

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

struct ForceState <: GMState{ForceGM}
    walls::SVector{4, Wall}
    objects::Vector{Dot}
end

get_objects(st::ForceState) = st.objects

function ForceState(gm::ForceGM, objects::AbstractVector{<:Dot})
    walls = init_walls(gm.area_width)
    ForceState(walls, objects)
end

function tracker_bounds(gm::ForceGM)
    @unpack area_width, area_height, dot_radius = gm
    xs = (-0.5*area_width + dot_radius, 0.5*area_width - dot_radius)
    ys = (-0.5*area_height + dot_radius, 0.5*area_height - dot_radius)
    (xs, ys)
end

function Dot(gm::ForceGM,
             pos::SVector{2, Float64},
             vel::SVector{2, Float64},
             target::Bool)
    t_dot = _Dot(gm.dot_radius, 1.0, pos, vel,
                gm.k_tail, target)
    update_graphics(gm, t_dot)
end

function step(gm::ForceGM,
              state::ForceState,
              forces::AbstractVector{<:SVector{2, Float64}})

    n_dots = gm.n_dots
    n_dots == length(forces) ||
        error("Length of `forces` missmatch with objects")

    objects = state.objects
    new_dots = Vector{Dot}(undef, n_dots)
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
        # kinematics: resolve forces to pos vel
        ku = update_kinematics(gm, dot, facc)
        new_dots[i] = update_graphics(gm, sync_update(dot, ku))
    end
    setproperties(state; objects = new_dots)
end

function step(gm::ForceGM,
              state::ForceState)

    # Dynamics (computing forces)
    n_dots = gm.n_dots
    @unpack walls, objects = state
    new_dots = Vector{Dot}(undef, n_dots)

    @inbounds for i in eachindex(objects)
        dot = objects[i]
        # force accumalator
        facc = MVector{2, Float64}(zeros(2))
        # interactions with walls
        for w in walls
            force!(facc, gm, w, dot)
        end
        for j in eachindex(objects)
            i === j && continue
            force!(facc, gm, objects[j], dot)
        end
        # kinematics: resolve forces to pos vel
        ku = update_kinematics(gm, dot, facc)
        new_dots[i] = sync_update(dot, ku)
    end
    setproperties(; objects = new_dots)
end

function force!(f::MVector{2, Float64}, gm::ForceGM, w::Wall, d::Dot)
    pos = get_pos(d)
    @unpack wall_repulsion, max_distance, distance_factor = gm
    v = LinearAlgebra.norm(w.normal .* pos + w.nd)
    v > max_distance && return nothing
    mag = wall_repulsion * exp(-v/distance_factor)
    f .+= mag * w.normal
    return nothing
end

function force!(f::MVector{2, Float64}, gm::ForceGM, x::Dot, d::Dot)
    @unpack dot_repulsion, max_distance, distance_factor = gm
    v = get_pos(d) - get_pos(x)
    nv = norm(v)
    nv > max_distance && return nothing
    mag = dot_repulsion * exp(-nv/distance_factor)
    delta_f = mag * (v./nv)
    f .+= delta_f
    return nothing
end

function update_kinematics(gm::ForceGM, d::Dot, f::MVector{2, Float64})
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
    # any(isnan, new_pos) && error()
    # setproperties(d; pos = new_pos, vel = v_adj)
    KinematicsUpdate(new_pos, new_vel)
end

function update_graphics(gm::ForceGM, d::Dot)

    nt = length(d.tail)

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

function rf_elements(gm::ForceGM, objects::AbstractVector{Dot})
    n = length(objects)
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n)
    # the trackers
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp, (obj.gstate,))
    end
    return es
end


function predict(gm::ForceGM, st::ForceState)
    rf_elements(gm, st.objects)
end

function observe(gm::ForceGM,
                 objects::AbstractVector{Dot})
    es = rf_elements(gm, objects)
    (es, gpp_mrfs(es, 50, 1.0))
end

include("gen.jl")

gen_fn(::ForceGM) = gm_force

################################################################################
# Misc
################################################################################

function objects_from_positions(gm::ForceGM, positions, targets)
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

function state_from_positions(gm::ForceGM, positions, targets)
    targets = collect(Bool, targets) # this is sometimes Int
    nt = length(positions)
    states = Vector{ForceState}(undef, nt)
    for t = 1:nt
        if t == 1
            dots = objects_from_positions(gm, positions[t], targets)
            states[t] = ForceState(gm, dots)
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
        states[t] = ForceState(gm, new_dots)
    end
    return states
end

function extract_rfs_subtrace(trace::Gen.Trace, t::Int64)
    # StaticIR names and nodes
    outer_ir = Gen.get_ir(gm_force)
    kernel_node = outer_ir.call_nodes[2] # (:kernel)
    kernel_field = Gen.get_subtrace_fieldname(kernel_node)
    # subtrace for each time step
    vector_trace = getproperty(trace, kernel_field)
    sub_trace = vector_trace.subtraces[t]
    # StaticIR for `force_kernel`
    inner_ir = Gen.get_ir(force_kernel)
    xs_node = inner_ir.call_nodes[2] # (:masks)
    xs_field = Gen.get_subtrace_fieldname(xs_node)
    # `RFSTrace` for :masks
    getproperty(sub_trace, xs_field)
end

const ForceGMIr = Gen.get_ir(gm_force)
const ForceGMTrace = Gen.get_trace_type(gm_force)

function td_flat(trace::ForceGMTrace,
                 ne::Int64,
                 temp::Float64)
    t, _... = get_args(trace)
    rfs = extract_rfs_subtrace(trace, t)
    pt = rfs.ptensor
    nx,_,np = size(pt)
    # ls::Float64 = logsumexp(pls)
    nls = log.(softmax(rfs.pscores, t=temp))
    # probability that each observation
    # is explained by a target
    x_weights = Vector{Float64}(undef, nx)
    @inbounds for x = 1:nx
        xw = -Inf
        @views for p = 1:np, e = 1:ne
            pt[x, e, p] || continue
            xw = logsumexp(xw, nls[p])
        end
        x_weights[x] = xw
    end

    # x_weights

    # # @show length(pls)
    # # display(sum(pt; dims = 3))
    # @show x_weights
    # the ratio of observations explained by each target
    # weighted by the probability that the observation is
    # explained by other targets
    td_weights = fill(-Inf, ne)
    @inbounds for i = 1:ne
        for p = 1:np
            ew = -Inf
            @views for x = 1:nx
                pt[x, i, p] || continue
                ew = x_weights[x]
                # assuming isomorphicity
                # (one association per partition)
                break
            end
            # P(e -> x) where x is associated with any other targets
            prop = nls[p]
            ew += prop
            td_weights[i] = logsumexp(td_weights[i], ew)
        end
        td_weights[i] =  clamp(round(td_weights[i]; digits=6), -Inf, 0.0)
    end
    # @show td_weights
    return td_weights
end
