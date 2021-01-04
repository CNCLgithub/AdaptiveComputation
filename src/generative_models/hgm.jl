using LinearAlgebra

@with_kw struct HGMParams <: AbstractGMParams
    n_trackers::Int = 4
    distractor_rate::Real = 4.0
    init_pos_spread::Real = 300.0
    polygon_radius::Real = 130.0
    
    # graphics parameters
    dot_radius::Real = 20.0
    img_height::Int = 200
    img_width::Int = 200
    area_height::Int = 800
    area_width::Int = 800

    # parameters for the drawing the mask random variable arguments
    dot_p::Float64 = 0.5 # prob of pixel on in the dot region
    gauss_amp::Float64 = 0.5 # gaussian amplitude for the gaussian component of the mask
    gauss_std::Float64 = 2.5 # standard deviation --||--

    # flow masks
    fmasks::Bool = false
    fmasks_decay_function::Function = default_decay_function
    fmasks_n = 5

    # probes
    probe_flip::Float64 = 0.0

    targets::Vector{Bool} = zeros(8)
end

function load(::Type{HGMParams}, path; kwargs...)
    HGMParams(;read_json(path)..., kwargs...)
end

function load(::Type{HGMParams}, path::String)
    HGMParams(;read_json(path)...)
end

const default_hgm = HGMParams()


@dist function sample_n_dots()
    categorical(fill(1.0/3, 3)) + 2
end

##### INIT STATE ######
@gen function sample_init_dot_or_polygon(gmh)::Object
    
    x = @trace(uniform(-gmh.init_pos_spread, gmh.init_pos_spread), :x)
    y = @trace(uniform(-gmh.init_pos_spread, gmh.init_pos_spread), :y)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)
    
    # sampling whether we're dealing with a polygon :O
    pol = @trace(bernoulli(0.5), :polygon)

    if pol
        r = gmh.polygon_radius
        rot = @trace(uniform(0, 2*pi), :rot)

        # 3, 4 or 5 dots in the polygon
        n_dots = @trace(sample_n_dots(), :n_dots)
        
        dots = Vector{Dot}(undef, n_dots)
        for i=1:n_dots
            # creating dots along the polygon
            dot_x = x + r * cos(2*pi*i/n_dots + rot)
            dot_y = y + r * sin(2*pi*i/n_dots + rot)
            #dot_x = x + r * cos(2*pi*i/n_dots)
            #dot_y = y + r * sin(2*pi*i/n_dots)
            
            # sprinkling some noise
            dot_x = @trace(normal(dot_x, 5.0), i => :x)
            dot_y = @trace(normal(dot_y, 5.0), i => :y)

            dots[i] = Dot([dot_x,dot_y,z], [0.0,0.0])
        end

        return Polygon([x,y,z], [0.0,0.0], rot, 0.0, r, dots)
    else
        return Dot([x,y,z], [0.0,0.0])
    end
end


init_dot_or_polygon_map = Gen.Map(sample_init_dot_or_polygon)

@gen function sample_init_hierarchical_state(hgm::HGMParams)
    hgm_trackers = fill(hgm, hgm.n_trackers)
    trackers = @trace(init_dot_or_polygon_map(hgm_trackers), :trackers)
    trackers = collect(Object, trackers)
    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)

    pmbrfs = RFSElements{Array}(undef, 0)

    if hgm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, hgm.n_trackers, hgm.fmasks_n)
        for i=1:hgm.n_trackers
            for j=1:hgm.fmasks_n
                fmasks[i,j] = zeros(hgm.img_height, hgm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               hgm.fmasks_decay_function)
    else
        flow_masks = nothing
    end

    FullState(graph, pmbrfs, flow_masks)

    return FullState(graph, pmbrfs, flow_masks)
end


@gen function hgm_pos_kernel(t::Int,
                            prev_state::FullState,
                            dynamics_model::AbstractDynamicsModel,
                            hgm::HGMParams)
    prev_graph = prev_state.graph
    new_graph = @trace(hgm_update(dynamics_model, prev_graph, hgm), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end

hgm_pos_chain = Gen.Unfold(hgm_pos_kernel)

@gen function hgm_pos(k::Int, motion::AbstractDynamicsModel,
                      hgm::HGMParams)
    
    init_state = @trace(sample_init_hierarchical_state(hgm), :init_state)
    states = @trace(hgm_pos_chain(k, init_state, motion, hgm), :kernel)
    result = (init_state, states)
    return result
end

function get_hgm_positions(cg::CausalGraph, targets::Vector{Bool})
    positions = []
    for e in cg.elements
        if isa(e, Polygon)
            positions = [positions; map(d -> d.pos, e.dots)]
        else
            push!(positions, e.pos)
        end
    end

    positions = positions[targets]
    return positions
end


@gen function hgm_mask_kernel(t::Int,
                                     prev_state::FullState,
                                     dynamics_model::AbstractDynamicsModel,
                                     hgm::HGMParams)
    
    prev_graph = prev_state.graph
    new_graph = @trace(hgm_update(dynamics_model, prev_graph, hgm), :dynamics)
    positions = get_hgm_positions(new_graph, hgm.targets)
    pmbrfs, flow_masks = get_masks_params(positions, hgm,
                                          flow_masks=prev_state.flow_masks)
    @trace(rfs(pmbrfs), :masks)

    new_state = FullState(new_graph, pmbrfs, flow_masks)
    return new_state
end

hgm_mask_chain = Gen.Unfold(hgm_mask_kernel)

@gen function hgm_mask(k::Int, motion::AbstractDynamicsModel,
                              hgm::HGMParams)
    init_state = @trace(sample_init_hierarchical_state(hgm), :init_state)
    states = @trace(hgm_mask_chain(k, init_state, motion, hgm), :kernel)
    result = (init_state, states)
    return result
end

export hgm_pos, hgm_mask, HGMParams, default_hgm
