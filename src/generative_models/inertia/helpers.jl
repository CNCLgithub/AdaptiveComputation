################################################################################
# Show helpers
################################################################################

import Base.show

################################################################################
# Gen helpers
################################################################################

function tracker_bounds(gm::InertiaGM)
    @unpack area_width, area_height, dot_radius = gm
    xs = (-0.5*area_width + dot_radius, 0.5*area_width - dot_radius)
    ys = (-0.5*area_height + dot_radius, 0.5*area_height - dot_radius)
    (xs, ys)
end

function InertiaState(prev_st::InertiaState,
                      new_dots,
                      es::RFSElements{T},
                      xs::Vector{T}) where {T}
    (pls, pt) = GenRFS.massociations(es, xs, 200, 1.)
    # (pls, pt) = GenRFS.associations(es, xs)
    setproperties(prev_st,
                  (objects = new_dots,
                   es = es,
                   xs = xs,
                   pt = pt,
                   pls = pls))
end

function InertiaState(gm::InertiaGM, dots)
    walls = init_walls(gm.area_width)
    n_ens = Float64(gm.n_dots - length(dots)) + 0.01
    InertiaState(walls, dots, UniformEnsemble(gm, n_ens),
                 RFSElements{BitMatrix}(undef, 0),
                 BitMatrix[],
                 falses(0,0,0),
                 Float64[])
end

function inertia_step_args(gm::InertiaGM, st::InertiaState)
    objs = get_objects(st)
    (fill(gm, length(objs)), objs)
end

# initializes a new dot
function Dot(gm::InertiaGM,
             pos::SVector{2, Float64},
             vel::SVector{2, Float64},
             target::Bool)
    t_dot = _Dot(gm.dot_radius, gm.dot_mass, pos, vel,
                gm.k_tail, target)
    update_graphics(gm, t_dot)
end


function sync_update(d::Dot,
                     ku::KinematicsUpdate)
    cb = deepcopy(d.tail)
    pushfirst!(cb, ku.p)
    setproperties(d, (vel = ku.v, tail = cb))
end

function UniformEnsemble(gm::InertiaGM,
                         rate::Float64)
    @unpack img_width, outer_f, inner_p, inner_f = gm
    # assuming square
    r = ceil(gm.dot_radius * inner_f * img_width / gm.area_width)
    n_pixels = prod(gm.img_dims)
    # pixel_prob = 1.7 * (pi * r^2 * inner_p) / n_pixels
    pixel_prob = 1.7 * (pi * r^2 * inner_p) / n_pixels
    UniformEnsemble(rate, pixel_prob, 0)
end

################################################################################
# RFS marginals
################################################################################

# function world(s::InertiaState)::CausalGraph
#     s.world
# end

# function assocs(st::InertiaState)
#     (st.pt, st.pls)
# end

function get_objects(st::InertiaState)
    st.objects
end

"""
Defines the `InertiaState` correspondence as a marginal
across partitions on non-zero target trackers.
"""
function correspondence(st::InertiaState)
    @unpack objects, ensemble, es, pt, pls = st
    targets = 1:4
    pt = pt[:, targets, :]
    correspondence(pt, pls)
end

function td_assocs(st::InertiaState)
    @unpack pt, pls = st
    ne = 4
    np = size(pt, 3)
    ws = exp.(pls .- logsumexp(pls))
    x_weights = Vector{Float64}(undef, 4)
    # first 4 objects are targets
    @inbounds @views for x = 1:4
        xw = 0.0
        for p = 1:np, e = 1:ne
            pt[x, e, p] || continue
            xw += ws[p]
        end
        x_weights[x] = xw
    end
    return x_weights
end

function td_flat(st::InertiaState, t::Float64)
    @unpack pt, pls = st
    nx,ne,np = size(pt)
    ne -= 1
    ls::Float64 = logsumexp(pls)
    nls = log.(softmax(pls, t=t))
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

    # the ratio of observations explained by each target
    # weighted by the probability that the observation is
    # explained by other targets
    td_weights = fill(-Inf, ne)
    @inbounds for i = 1:ne
        # @show i
        for p = 1:np
            ew = -Inf
            @views for x = 1:nx
                pt[x, i, p] || continue
                ew = x_weights[x]
                # println("pls($p) = $(pls[p]), xw($x) = $ew")
                # assuming isomorphicity
                # (one association per partition)
                break
            end
            # P(e -> x) where x is associated with any other targets
            prop = nls[p] # ((pls[p]/t - ls/t))
            ew += prop
            # println("pls($p) corrected = $(exp(prop))")
            td_weights[i] = logsumexp(td_weights[i], ew)
        end
    end
    # @show pls
    # @show x_weights
    # # display(exp.(x_weights))
    # @show td_weights
    # # display(exp.(td_weights))
    # error()
    return td_weights
end

function td_full(st::InertiaState)
    @unpack es, pt, pls = st
    # all trackers (anything that isnt an ensemble)
    tracker_ids = findall(x -> isa(x, BernoulliElement), es)
    pt = pt[:, tracker_ids, :]
    td_full(pt, pls) # P({x...} are targets)
end


function target_weights(st::InertiaState, wv::Vector{Float64})
    c = correspondence(st)
    ne = size(c, 2)
    tws = zeros(ne)
    @inbounds for ti = 1:ne
        tws[ti] = sum(c[:, ti] .* wv)
    end
    return tws
end

function trackers(dm::InertiaGM, tr::Trace)
    t = first(get_args(tr))
    st = tr[:kernel => t]
    n = length(st.objects)
    ts = Vector{Pair}(undef, n)
    @inbounds for i = 1:n
        ts[i] = :kernel =>  t => :dynamics => :trackers => i
    end
    ts
end

################################################################################
# Misc
################################################################################

function objects_from_positions(gm::InertiaGM, positions)
    nx = length(positions)
    dots = Vector{Dot}(undef, nx)
    for i = 1:nx
        xy = Float64.(positions[i][1:2])
        dots[i] = Dot(gm,
                      SVector{2}(xy),
                      SVector{2, Float64}(zeros(2)),
                      false)
    end
    return dots
end

function state_from_positions(gm::InertiaGM, positions, targets)
    nt = length(positions)
    states = Vector{InertiaState}(undef, nt)
    for t = 1:nt
        if t == 1
            dots = objects_from_positions(gm, positions[t])
            states[t] = InertiaState(gm, dots)
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
        es, xs = observe(gm, new_dots)
        states[t] = InertiaState(prev_state,
                                 new_dots,
                                 es,
                                 xs)
    end
    return states
end

function load_scene(gm::InertiaGM, dataset_path::String, scene::Int64)
    scene_data = JSON.parsefile(dataset_path)[scene]
    aux_data = scene_data["aux_data"]
    states = state_from_positions(gm,
                                  scene_data["positions"],
                                  aux_data["targets"])
    scene_data = Dict(:gt_states => states,
                       :aux_data => aux_data)
end
