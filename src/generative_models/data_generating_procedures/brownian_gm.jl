export dgp_gm


"""
    data generating procedure
    that takes the masks produced from
    the Brownian generative model directly
"""

function dgp_gm(k::Int, gm::GMMaskParams,
             motion::BrownianDynamicsModel)

    trace, _ = Gen.generate(gm_masks_static, (k, motion, gm))

    masks = Vector{Vector{BitArray{2}}}(undef, k)
    for t=1:k
        masks[t] = trace[:states => t => :masks]
    end

    init_positions = Array{Float64}(undef, gm.n_trackers, 3)
    init_state, _ = Gen.get_retval(trace)

    for i=1:gm.n_trackers
        init_positions[i,:] = init_state.graph.elements[i].pos
    end

    return init_positions, masks
end
