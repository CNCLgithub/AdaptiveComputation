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
    (pls, pt) = GenRFS.massociations(es, xs, 50, 1.0)
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
                 RFSElements{Any}(undef, 0),
                 [],
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
    t_dot = Dot(gm.dot_radius, gm.dot_mass, pos, vel,
                target, spzeros(gm.img_dims))
    gs = update_graphics(gm, t_dot, pos)
    update(t_dot, pos, vel, gs)
end


function update(d::Dot,
                pos::SVector{2, Float64},
                vel::SVector{2, Float64},
                gstate)
    setproperties(d,
                  (pos = pos, vel = vel, gstate = gstate))
end

function UniformEnsemble(gm::InertiaGM,
                         rate::Float64)
    @unpack img_width, outer_f, inner_p, inner_f = gm
    # assuming square
    r = ceil(gm.dot_radius * inner_f * img_width / gm.area_width)
    n_pixels = prod(gm.img_dims)
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

function td_flat(st::InertiaState)
    @unpack pt, pls = st
    targets = 1:4
    pt = pt[:, targets, :]
    nx,ne,np = size(pt)
    amat = zeros(nx, ne)
    ws = exp.(pls .- logsumexp(pls))
    @inbounds @views for p = 1:np, e = 1:ne, x = 1:nx
        pt[x, e, p] || continue
        amat[x, e] += ws[p]
    end
    x_weights = sum(eachcol(amat)) # P(x_i = Target)
    # @show np
    # display(amat)
    # display(x_weights)
    td_weights = zeros(4)
    @inbounds @views for i = targets
        for p = 1:np
            kx = 0
            xw = 0
            for x = 1:nx
                pt[x, i, p] || continue
                xw += x_weights[x]
                kx += 1
            end
            kx == 0 && continue
            # P(e -> x) where x is associated with any other targets
            td_weights[i] += (xw / kx) * ws[p]
        end
    end
    # @show td_weights
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



function exp_dot_mask( x0::Float64, y0::Float64,
                       r::Float64,
                       w::Int64, h::Int64,
                       gm::InertiaGM)
    exp_dot_mask(x0, y0, r, w, h,
                 gm.outer_f,
                 gm.inner_f,
                 gm.outer_p,
                 gm.inner_p)
end

# function render_from_cgs(states,
#                          gm::GMParams,
#                          cgs::Vector{CausalGraph})
#     k = length(cgs)
#     # time x thing
#     # first time step is initialization (not inferred)
#     bit_masks= Vector{Vector{BitMatrix}}(undef, k-1)

#     # initialize graphics
#     g = first(cgs)
#     set_prop!(g, :gm, gm)
#     gr_diff = render(gr, g)
#     @inbounds for t = 2:k
#         g = cgs[t]
#         set_prop!(g, :gm, gm)
#         # carry over graphics from last step
#         patch!(g, gr_diff)
#         # render graphics from current step
#         gr_diff = render(gr, g)
#         patch!(g, gr_diff)
#         @>> g predict(gr) patch!(g)
#         # create masks
#         bit_masks[t - 1] = rfs(get_prop(g, :es))
#     end
#     bit_masks
# end

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
        ni = length(prev_state.objects)
        kus = Vector{KinematicsUpdate}(undef, ni)
        for i = 1:ni
        old_pos = Float64.(positions[t-1][i][1:2])
        new_pos = Float64.(positions[t][i][1:2])
        new_vel = (new_pos .- old_pos)
            kus[i] = KinematicsUpdate(SVector{2}(new_pos),
                                      SVector{2}(new_vel))
        end
        new_dots = step(gm, prev_state, kus)
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
