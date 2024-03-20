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

"helper used to initialize `InertiaState` in `inertia_init`"
function InertiaState(gm::InertiaGM, dots)
    walls = init_walls(gm.area_width)
    n_ens = Float64(gm.n_dots - length(dots)) + 0.01
    InertiaState(walls, dots, UniformEnsemble(gm, n_ens))
                 # RFSElements{BitMatrix}(undef, 0),
                 # BitMatrix[],
                 # falses(0,0,0),
                 # Float64[])
end

"creates next state, used in `inertia_kernel`"
function InertiaState(prev_st::InertiaState,
                      new_dots)
                      # es::RFSElements{T},
                      # xs::Vector{T}) where {T}
    # (pls, pt) = GenRFS.massociations(es, xs, 200, 1.)
    setproperties(prev_st,
                  (objects = new_dots))
                  # (objects = new_dots,
                  #  es = es,
                  #  xs = xs,
                  #  pt = pt,
                  #  pls = pls))
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

# """
# Defines the `InertiaState` correspondence as a marginal
# across partitions on non-zero target trackers.
# """
# function correspondence(st::InertiaState)
#     @unpack objects, ensemble, es, pt, pls = st
#     targets = Int64(sum(map(MOT.target, objects)))
#     pt = pt[:, targets, :]
#     correspondence(pt, pls)
# end

"The probability in `st` that each gt target is associated with any target"
function td_assocs(st::InertiaState)
    @unpack pt, pls = st
    nx,ne,np = size(pt)
    num_targets = ne - 1
    np = size(pt, 3) # number of partitions
    # normalized weights of each partition
    pws = exp.(pls .- logsumexp(pls))
    weights = Vector{Float64}(undef, num_targets)
    # first 4 objects / observations are targets in gt
    @inbounds @views for x = 1:num_targets
        w = 0.0 # Pr(x_i -> e_{1, N})
        for p = 1:np, e = 1:num_targets
            pt[x, e, p] || continue
            w += pws[p]
        end
        weights[x] = w
    end
    return weights
end

function ensemble_uncertainty(st::InertiaState,
                              t::Float64)
    @unpack pt, pls = st
    nx,ne,np = size(pt)
    ls::Float64 = logsumexp(pls)
    nls = log.(softmax(pls, t=t))
    # probability that each observation
    # is explained by a target
    x_weights = Vector{Float64}(undef, nx)
    @inbounds for x = 1:nx
        xw = -Inf
        @views for p = 1:np
            # assigned to ensemble
            pt[x, ne, p] || continue
            xw = logsumexp(xw, nls[p])
        end
        x_weights[x] = xw
    end
    return x_weights
end


function td_full(st::InertiaState)
    @unpack es, pt, pls = st
    # all trackers (anything that isnt an ensemble)
    tracker_ids = findall(x -> isa(x, BernoulliElement), es)
    pt = pt[:, tracker_ids, :]
    td_full(pt, pls) # P({x...} are targets)
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

function objects_from_positions(gm::InertiaGM, positions, targets)
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

function state_from_positions(gm::InertiaGM, positions, targets)
    targets = collect(Bool, targets) # this is sometimes Int
    nt = length(positions)
    states = Vector{InertiaState}(undef, nt)
    for t = 1:nt
        if t == 1
            dots = objects_from_positions(gm, positions[t], targets)
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
        # es, xs = observe(gm, new_dots)
        states[t] = InertiaState(prev_state,
                                 new_dots)
                                 # es,
                                 # xs)
    end
    return states
end

function load_scene(gm::GenerativeModel, scene_data::Dict)
    aux_data = scene_data["aux_data"]
    states = state_from_positions(gm,
                                  scene_data["positions"],
                                  aux_data["targets"])
    Dict(:gt_states => states,
         :aux_data => aux_data)
end

function load_scene(gm::GenerativeModel, dataset_path::String, scene::Int64)
    scene_data = JSON.parsefile(dataset_path)[scene]
    load_scene(gm, scene_data)
end
