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


struct InertiaState <: GMState{InertiaGM}
    walls::SVector{4, Wall}
    objects::Vector{Dot}
    ensemble::UniformEnsemble
    # es::RFSElements
    # xs::AbstractArray
    # pt::BitArray{3}
    # pls::Vector{Float64}
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
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n + 1)
    # es = RFSElements{GaussObs{2}}(undef, n + 1)
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
    # es = RFSElements{GaussObs{2}}(undef, n)
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n)
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp, (obj.gstate,))
    end
    (es, gpp_mrfs(es, 50, 1.0))
end

include("helpers.jl")
include("gen.jl")

gen_fn(::InertiaGM) = gm_inertia
const InertiaIr = Gen.get_ir(gm_inertia)
const InertiaTrace = Gen.get_trace_type(gm_inertia)

function extract_rfs_subtrace(trace::InertiaTrace, t::Int64)
    # StaticIR names and nodes
    outer_ir = Gen.get_ir(gm_inertia)
    kernel_node = outer_ir.call_nodes[2] # (:kernel)
    kernel_field = Gen.get_subtrace_fieldname(kernel_node)
    # subtrace for each time step
    vector_trace = getproperty(trace, kernel_field)
    sub_trace = vector_trace.subtraces[t]
    # StaticIR for `inertia_kernel`
    inner_ir = Gen.get_ir(inertia_kernel)
    xs_node = inner_ir.call_nodes[2] # (:masks)
    xs_field = Gen.get_subtrace_fieldname(xs_node)
    # `RFSTrace` for :masks
    getproperty(sub_trace, xs_field)
end

function td_flat(trace::InertiaTrace, temp::Float64)

    t = first(get_args(trace))
    rfs = extract_rfs_subtrace(trace, t)
    pt = rfs.ptensor
    # @unpack pt, pls = st
    nx,ne,np = size(pt)
    ne -= 1
    # ls::Float64 = logsumexp(pls)
    nls = log.(softmax(rfs.pscores, t=temp))
    # probability that each observation
    # is explained by a target
    x_weights = Vector{Float64}(undef, nx)
    @inbounds for x = 1:nx
        xw = -Inf
        @views for p = 1:np
            if !pt[x, ne + 1, p]
                xw = logsumexp(xw, nls[p])
            end
        end
        x_weights[x] = xw
    end

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
    end
    return td_weights
end

function scan_ptensor(pt, e::Int, p::Int, nx::Int)
    idx = 0
    @views for x = 1:nx
        if pt[x, e, p]
            idx = x
            break
        end
    end
    return idx
end

function id_flat(trace::InertiaTrace, temp::Float64)

    t = first(get_args(trace))
    rfs = extract_rfs_subtrace(trace, t)
    pt = rfs.ptensor
    nx,ne,np = size(pt)
    ne -= 1
    ls = softmax(rfs.pscores, t=temp)
    # identity confidence
    ws = Vector{Float64}(undef, ne)
    hs = Vector{Float64}(undef, nx)
    @inbounds for e = 1:ne
        fill!(hs, -Inf)
        for p = 1:np
            x = scan_ptensor(pt, e, p, nx)
            hs[x] = logsumexp(hs[x], ls[p])
        end
        ws[e] = maximum(hs)
    end
    return ws
end
