@gen static function isr_pos_kernel(t::Int,
                                prev_state::FullState,
                                dynamics_model::AbstractDynamicsModel,
                                params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(isr_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end
isr_pos_chain = Gen.Unfold(isr_pos_kernel)

@gen static function isr_mask_kernel(t::Int,
                                 prev_state::FullState,
                                 dynamics_model::AbstractDynamicsModel,
                                 params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(isr_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs,fm = get_masks_params(new_trackers, params)
    @trace(rfs(pmbrfs), :masks)
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end
isr_mask_chain = Gen.Unfold(isr_mask_kernel)


@gen static function gm_isr_pos(T::Int, motion::AbstractDynamicsModel,
                                params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(isr_pos_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states, nothing)
    return result
end

@gen static function gm_isr_mask(T::Int, motion::AbstractDynamicsModel,
                                 params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(isr_mask_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states, nothing)
    return result
end

export gm_isr_mask, gm_isr_pos
