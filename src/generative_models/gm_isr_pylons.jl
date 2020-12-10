#@gen static function isr_pylons_pos_kernel(t::Int,
@gen function isr_pylons_pos_kernel(t::Int,
                                prev_state::FullState,
                                dynamics_model::AbstractDynamicsModel,
                                params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(isr_pylons_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end
isr_pylons_pos_chain = Gen.Unfold(isr_pylons_pos_kernel)


# #@gen (static) function sample_init_tracker(init_pos_spread::Real)::Dot
@gen function sample_init_tracker_pylons(init_pos_spread::Real, exp0::Bool)::Dot

    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)

    # initial velocity is zero (in the new version)
    vx = 0.0
    vy = 0.0

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)
    
    idx = @trace(Gen.categorical([0.5, 0.0, 0.5]), :pylon_interaction)
    pylon_interaction = [-1,0,1][idx]
    #pylon_interaction = 1

    # initial velocity is zero
    return Dot(pos=[x,y,z], vel=[vx, vy],
               pylon_interaction=pylon_interaction)
end

init_trackers_pylons_map = Gen.Map(sample_init_tracker_pylons)

@gen function sample_init_isr_pylons_state(gm::GMMaskParams, motion::ISRPylonsDynamics)
    trackers_gm = fill(gm.init_pos_spread, gm.n_trackers)
    exp0 = fill(gm.exp0, gm.n_trackers)
    trackers = @trace(init_trackers_pylons_map(trackers_gm, exp0), :trackers)
    trackers = collect(Dot, trackers)

    # pylons
    combs = collect.(Iterators.product([-1,1], [-1,1]))
    pylon_positions = vec(combs .* [motion.pylon_x, motion.pylon_y])
    pylons = map(pos -> Pylon([pos; 1.0], motion.pylon_radius, motion.pylon_strength), pylon_positions)

    println(pylons)

    # add each tracker and pylon to the graph as independent vertices
    elements = [trackers; pylons]
    graph = CausalGraph(elements, SimpleGraph)
    pmbrfs = RFSElements{Array}(undef, 0)

    if gm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, gm.n_trackers, gm.fmasks_n)
        for i=1:gm.n_trackers
            for j=1:gm.fmasks_n
                fmasks[i,j] = zeros(gm.img_height, gm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               gm.fmasks_decay_function)
    else
        flow_masks = nothing
    end

    FullState(graph, pmbrfs, flow_masks)
end

#@gen static function isr_pylons_mask_kernel(t::Int,
@gen function isr_pylons_mask_kernel(t::Int,
                                 prev_state::FullState,
                                 dynamics_model::AbstractDynamicsModel,
                                 params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(isr_pylons_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs, flow_masks = get_masks_params(new_trackers, params, flow_masks=prev_state.flow_masks)
    @trace(rfs(pmbrfs), :masks)
    new_state = FullState(new_graph, pmbrfs, flow_masks)
    return new_state
end
isr_pylons_mask_chain = Gen.Unfold(isr_pylons_mask_kernel)


#@gen static function gm_isr_pylons_pos(T::Int, motion::AbstractDynamicsModel,
@gen function gm_isr_pylons_pos(T::Int, motion::AbstractDynamicsModel,
                                params::GMMaskParams)
    init_state = @trace(sample_init_isr_pylons_state(params, motion), :init_state)
    states = @trace(isr_pylons_pos_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states, nothing)
    return result
end

#@gen static function gm_isr_pylons_mask(T::Int, motion::AbstractDynamicsModel,
@gen function gm_isr_pylons_mask(T::Int, motion::AbstractDynamicsModel,
                                 params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(isr_pylons_mask_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states, nothing)
    return result
end

export gm_isr_pylons_mask, gm_isr_pylons_pos
