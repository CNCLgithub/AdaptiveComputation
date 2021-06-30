
# @gen function inertia_kernel(t::Int,
@gen static function inertia_kernel(t::Int,
                                    prev_cg::CausalGraph)

    # advancing causal graph according to dynamics
    # (there is a deepcopy here)
    cg = @trace(inertial_update(prev_cg), :dynamics) 

    # updating graphics + getting parameters for the prediction
    cg_rfs = graphics_update(cg)
    rfs_vec = get_prop(cg_rfs, :rfs_vec)
    @trace(Map(sample_masks)(rfs_vec), :receptive_fields)

    return cg_rfs
end

@gen static function gm_inertia_mask(k::Int,
# @gen function gm_inertia_mask(k::Int,
                              gm, dm, graphics)
    
    cg = get_init_cg(gm, dm, graphics)
    init_state = @trace(sample_init_state(cg), :init_state)
    states = @trace(Gen.Unfold(inertia_kernel)(k, init_state), :kernel)
    result = (init_state, states)
    return result
end

export gm_inertia_mask
