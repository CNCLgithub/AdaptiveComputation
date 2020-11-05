
@gen static function inertia_kernel(t::Int,
                                 prev_state::FullState,
                                 dynamics_model::AbstractDynamicsModel,
                                 params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(inertial_update(dynamics_model, prev_graph), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs, flow_masks = get_masks_params(new_trackers, params,
                                          flow_masks=prev_state.flow_masks)
    @trace(rfs(pmbrfs), :masks)
    new_state = FullState(new_graph, pmbrfs, flow_masks)
    return new_state
end
inertia_chain = Gen.Unfold(inertia_kernel)

@gen static function gm_inertia_mask(T::Int, motion::AbstractDynamicsModel,
                                params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(inertia_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states, nothing)
    return result
end

export gm_inertia_mask
