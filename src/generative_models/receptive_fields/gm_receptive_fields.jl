
@gen static function receptive_fields_kernel(t::Int,
                                      prev_state::ReceptiveFieldsState,
                                      dynamics_model::AbstractDynamicsModel,
                                      receptive_fields::Vector{AbstractReceptiveField},
                                      prob_threshold::Float64,
                                      gm::GMParams)
    # using Inertia Dynamics
    new_cg = @trace(inertial_update(dynamics_model, prev_state.cg), :dynamics)
    objects = new_cg.elements
    rfs_vec, flow_masks = get_rfs_vec(receptive_fields, objects, prob_threshold, gm, flow_masks=prev_state.flow_masks)
    @trace(Map(sample_masks)(rfs_vec), :receptive_fields)

    # returning this to get target designation and assignment
    new_state = ReceptiveFieldsState(new_cg, rfs_vec, prev_state.flow_masks)
    return new_state
end

receptive_fields_chain = Gen.Unfold(receptive_fields_kernel)

@gen static function gm_receptive_fields(k::Int,
                                         dynamics_model::AbstractDynamicsModel,
                                         gm::GMParams,
                                         receptive_fields::Vector{AbstractReceptiveField},
                                         prob_threshold::Float64)
    init_state = @trace(sample_init_receptive_fields_state(gm), :init_state)
    states = @trace(receptive_fields_chain(k, init_state, dynamics_model, receptive_fields, prob_threshold, gm), :kernel)

    result = (init_state, states)
    return result
end


export gm_receptive_fields
