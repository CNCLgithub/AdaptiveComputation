export gm_masks_isr_static

using LinearAlgebra

##################################

@gen static function kernel(t::Int,
                            prev_state::FullState,
                            dynamics_model::AbstractDynamicsModel,
                            params::GMMaskParams)

    prev_graph = prev_state.graph

    new_graph = @trace(isr_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements

    pmbrfs = get_masks_params(new_trackers, params)
    # rfs_rec = AssociationRecord(params.record_size)
    @trace(rfs(pmbrfs), :masks)

    # returning this to get target designation and assignment
    # later (HACKY STUFF) saving as part of state
    new_state = FullState(new_graph, pmbrfs)

    return new_state
end

chain = Gen.Unfold(kernel)

@gen static function gm_masks_isr_static(T::Int, motion::AbstractDynamicsModel,
                                       params::GMMaskParams)
    
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(chain(T, init_state, motion, params), :states)

    result = (init_state, states)

    return result
end
