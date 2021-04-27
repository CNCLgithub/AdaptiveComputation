
@gen function inertia_kernel(t::Int,
                                 prev_state::RFState,
                                 dm::InertiaModel,
                                 gm::GMParams,
                                 receptive_fields::Vector{RectangleReceptiveField},
                                 prob_threshold::Float64)
    new_cg = @trace(inertial_update(dm, prev_state.cg), :dynamics)
    dots = @>> get_objects(new_cg, Dot)
    rfs_vec, flow_masks = get_rfs_vec(receptive_fields, dots, prob_threshold, gm,
                                      flow_masks=prev_state.flow_masks)
    @trace(Map(sample_masks)(rfs_vec), :receptive_fields)
    new_state = RFState(new_cg, rfs_vec, flow_masks)
    return new_state
end

inertia_chain = Gen.Unfold(inertia_kernel)

@gen function gm_inertia_mask(k::Int, dm::InertiaModel, gm::GMParams,
                                 receptive_fields::Vector{RectangleReceptiveField},
                                 prob_threshold::Float64)
    init_state = @trace(sample_init_receptive_fields_state(gm, dm), :init_state)
    states = @trace(inertia_chain(k, init_state, dm, gm,
                                  receptive_fields, prob_threshold), :kernel)
    result = (init_state, states, nothing)
    return result
end

export gm_inertia_mask
