import Base.show

uheatmap = UnicodePlots.heatmap
uspy = UnicodePlots.spy
rargs = GenRFS.args

function Base.show(io::IO, ::MIME"text/plain", v::InertiaState)
    println(io, "Elements")
    for i in eachindex(v.es)
        @>> (v.es[i]) begin
            rargs
            first
            (x -> x .> 0.)
            (x -> v.xs[i] .& (.!(x)))
            uspy
            println(io)
        end

    end
    # println(io, "Observations")
    # for i in eachindex(v.xs)
    #     @>> (v.xs[i]) begin
    #         uspy
    #         println(io)
    #     end
    # end
    return nothing
end

struct KinematicsUpdate
    p::SVector{2, Float64}
    v::SVector{2, Float64}
end

function tracker_bounds(gm::InertiaGM
    @unpack area_width, area_height, dot_radius = gm
    xs = (-0.5*area_width + dot_radius, 0.5*area_width - dot_radius)
    ys = (-0.5*area_height + dot_radius, 0.5*area_height - dot_radius)
    (xs, ys)
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


function world(s::InertiaState)::CausalGraph
    s.world
end

function assocs(st::InertiaState)
    (st.pt, st.pls)
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
    @unpack world, es, pt, pls = st
    targets = 1:4
    pt = pt[:, targets, :]
    td_flat(pt, pls) # P(x_i = Target)
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

function trackers(dm::InertiaModel, tr::Trace)
    t = first(get_args(tr))
    st = tr[:kernel => t]
    n = length(st.objects)
    ts = Vector{Pair}(undef, n)
    @inbounds for i = 1:n
        ts[i] = :kernel =>  t => :dynamics => :trackers => i
    end
    ts
end


function UniformEnsemble(gm::InertiaGM,
                         rate,
                         targets)
    @unpack img_width, outer_f, inner_p, inner_f = gm
    # assuming square
    r = ceil(gm.dot_radius * inner_f * img_width / gm.area_width)
    n_pixels = prod(gr.img_dims)
    pixel_prob = (pi * r^2 * inner_p) / n_pixels
    UniformEnsemble(rate, pixel_prob, targets)
end


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

function render_from_cgs(states,
                         gm::GMParams,
                         cgs::Vector{CausalGraph})
    k = length(cgs)
    # time x thing
    # first time step is initialization (not inferred)
    bit_masks= Vector{Vector{BitMatrix}}(undef, k-1)

    # initialize graphics
    g = first(cgs)
    set_prop!(g, :gm, gm)
    gr_diff = render(gr, g)
    @inbounds for t = 2:k
        g = cgs[t]
        set_prop!(g, :gm, gm)
        # carry over graphics from last step
        patch!(g, gr_diff)
        # render graphics from current step
        gr_diff = render(gr, g)
        patch!(g, gr_diff)
        @>> g predict(gr) patch!(g)
        # create masks
        bit_masks[t - 1] = rfs(get_prop(g, :es))
    end
    bit_masks
end

function cg_from_positions(positions, targets)
    nt = length(positions)
    cgs = Vector{CausalGraph}(undef, nt)
    for t = 1:nt
        g = CausalGraph()
        step_pos = positions[t]
        for j = 1:length(step_pos)
            d = Dot(pos = step_pos[j],
                    target = targets[j])
            add_vertex!(g)
            set_prop!(g, j, :object, d)
        end
        cgs[t] = g
    end
    cgs
end

"""
    loads gt_causal_graphs and aux_data
"""
function load_scene(dataset_path::String, scene::Int64)
    scene_data = JSON.parsefile(dataset_path)[scene]
    aux_data = scene_data["aux_data"]
    cgs = cg_from_positions(scene_data["positions"],
                            aux_data["targets"])
    scene_data = Dict(:gt_causal_graphs => cgs,
                       :aux_data => aux_data)
end
