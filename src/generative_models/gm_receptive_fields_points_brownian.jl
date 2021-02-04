
@gen static function receptive_fields_points_brownian_kernel(t::Int,
                                      prev_state::ReceptiveFieldsPointsState,
                                      dynamics_model::BrownianDynamicsModel,
                                      receptive_fields::Vector{RectangleReceptiveField},
                                      gm::GMPointParams)

    # using ISR Dynamics
    new_graph = @trace(brownian_update(dynamics_model, prev_state.graph), :dynamics)
    objects = new_graph.elements

    rfs_vec = get_rfs_vec_points(receptive_fields, objects, gm)
    # rfs_vec, flow_masks = get_rfs_vec(receptive_fields, objects, prob_threshold, gm, flow_masks=prev_state.flow_masks)
    @trace(receptive_fields_points_map(rfs_vec), :receptive_fields)

    # returning this to get target designation and assignment
    new_state = ReceptiveFieldsPointsState(new_graph, rfs_vec)
    return new_state
end

receptive_fields_points_brownian_chain = Gen.Unfold(receptive_fields_points_brownian_kernel)

@gen static function gm_receptive_fields_points_brownian(k::Int,
                                         dynamics_model::BrownianDynamicsModel,
                                         gm::GMPointParams,
                                         receptive_fields::Vector{RectangleReceptiveField})
    init_state = @trace(sample_init_receptive_fields_points_state(gm), :init_state)
    states = @trace(receptive_fields_points_brownian_chain(k, init_state, dynamics_model, receptive_fields, gm), :kernel)

    result = (init_state, states)
    return result
end


export gm_receptive_fields_points_brownian
