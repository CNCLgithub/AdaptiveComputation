
#@gen static function inertia_kernel(t::Int,
@gen function inertia_kernel(t::Int, prev_cg::CausalGraph)

    # advancing causal graph according to dynamics
    # (there is a deepcopy here)
    cg = @trace(inertial_update(prev_cg), :dynamics) 

    # updating graphics + getting parameters for the prediction
    rfs_vec = graphics_update!(cg)
    
    @trace(Map(sample_masks)(rfs_vec), :receptive_fields)

    return cg
end

#@gen static function gm_inertia_mask(k::Int,
@gen static function gm_inertia_mask(k::Int,
                                     gm::GMParams,
                                     dm::InertiaModel,
                                     graphics::Graphics)

    init_cg = @trace(sample_init_cg(gm, dm, graphics), :init_cg)
    cgs = @trace(Gen.Unfold(inertia_kernel)(k, init_cg), :kernel)
    result = (init_cg, cgs)
    return result
end

export gm_inertia_mask
