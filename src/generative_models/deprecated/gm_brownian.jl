
##################################
@gen static function br_pos_kernel(t::Int,
                            prev_state::State,
                            dynamics_model::AbstractDynamicsModel,
                            params::GMParams)
    prev_graph = prev_state.graph
    new_graph = @trace(brownian_update(dynamics_model, prev_graph), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = State(new_graph, pmbrfs, nothing)
    return new_state
end

br_pos_chain = Gen.Unfold(br_pos_kernel)

@gen static function gm_brownian_pos(T::Int, motion::AbstractDynamicsModel,
                                       params::GMParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(br_pos_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end

@gen static function br_mask_kernel(t::Int,
                            prev_state::State,
                            dynamics_model::AbstractDynamicsModel,
                            params::GMParams)

    prev_graph = prev_state.graph
    new_graph = @trace(brownian_update(dynamics_model, prev_graph), :dynamics)
    positions = map(e -> e.pos, new_graph.elements)
    pmbrfs, flow_masks = get_masks_params(positions, params, flow_masks=prev_state.flow_masks)
    @trace(rfs(pmbrfs), :masks)

    # returning this to get target designation and assignment
    new_state = State(new_graph, pmbrfs, flow_masks)

    return new_state
end

br_mask_chain = Gen.Unfold(br_mask_kernel)

@gen static function gm_brownian_mask(T::Int, motion::AbstractDynamicsModel,
#@gen function gm_brownian_mask(T::Int, motion::AbstractDynamicsModel,
                                       params::GMParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(br_mask_chain(T, init_state, motion, params), :kernel)

    result = (init_state, states)

    return result
end

export gm_brownian_pos, gm_brownian_mask, default_gm
