export dgp_gm


"""
    data generating procedure
    that takes the masks produced from
    the Brownian generative model directly
"""

function dgp_gm(k::Int, gm::GMMaskParams,
                motion::BrownianDynamicsModel)
    
    trace, _ = Gen.generate(gm_brownian_mask, (k, motion, gm))
    init_state, states = Gen.get_retval(trace)

    gt_causal_graphs = Vector{CausalGraph}(undef, length(states)+1)
    gt_causal_graphs[1] = init_state.graph
    gt_causal_graphs[2:end] = map(x->x.graph, states)

    masks = Vector{Vector{BitArray{2}}}(undef, k)
    for t=1:k
        masks[t] = trace[:kernel => t => :masks]
    end

    trial_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :motion => motion,
                       :masks => masks])
end
