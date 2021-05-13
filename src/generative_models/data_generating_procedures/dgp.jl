export dgp

using Setfield

_dgp(dm::SquishyDynamicsModel, k::Int, cg::CausalGraph, cm::ChoiceMap) = Gen.generate(squishy_gm_pos, (k, cg), cm)

function dgp(k::Int, cg::CausalGraph;
             generate_masks=true,
             generate_cm=false,
             cm::ChoiceMap=choicemap())
    
    # new params with all dots having state for data generation
    gm = get_gm(cg)
    gm = deepcopy(gm)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    cg = deepcopy(cg)
    set_prop!(cg, :gm, gm)

    # running generative model on just positions (no need to go to masks)
    dm = get_dm(cg)
    trace, _ = _dgp(dm, k, cg, cm)
    init_cg, cgs = Gen.get_retval(trace)

    gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
    gt_causal_graphs[1] = init_cg
    gt_causal_graphs[2:end] = cgs
    
    masks = generate_masks ? get_masks(gt_causal_graphs) : nothing
    cm = generate_cm ? Gen.get_choices(trace) : nothing
    
    scene_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :masks => masks,
                       :cm => cm])
end
