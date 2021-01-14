@gen function sample_masks(pmbrfs)
    @trace(rfs(pmbrfs), :masks) 
end

receptive_fields_map = Gen.Map(sample_masks)

@gen function receptive_fields_kernel(t::Int,
                                      prev_state::FullState,
                                      dynamics_model::AbstractDynamicsModel,
                                      receptive_fields::Vector{AbstractReceptiveField},
                                      prob_threshold::Float64,
                                      gm::GMMaskParams)
    # using ISR Dynamics
    new_graph = @trace(isr_update(dynamics_model, prev_state.graph, gm), :dynamics)
    objects = new_graph.elements
    rfs_vec = get_rfs_vec(receptive_fields, objects, prob_threshold, gm)
    @trace(receptive_fields_map(rfs_vec), :receptive_fields)

    # returning this to get target designation and assignment
    new_state = FullState(new_graph, prev_state.rfs, prev_state.flow_masks)
    return new_state
end

receptive_fields_chain = Gen.Unfold(receptive_fields_kernel)

@gen static function gm_receptive_fields(k::Int,
                                  dynamics_model::AbstractDynamicsModel,
                                  gm::GMMaskParams,
                                  receptive_fields::Vector{AbstractReceptiveField},
                                  prob_threshold::Float64)
    init_state = @trace(sample_init_state(gm), :init_state)
    states = @trace(receptive_fields_chain(k, init_state, dynamics_model, receptive_fields, prob_threshold, gm), :kernel)

    result = (init_state, states)
    return result
end

export gm_receptive_fields
