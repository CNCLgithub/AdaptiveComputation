export dgp

using Setfield

_dgp(k::Int, gm::GMMaskParams, motion::AbstractDynamicsModel, cm::ChoiceMap) = error("not implemented")
_dgp(k::Int, gm::GMMaskParams, motion::BrownianDynamicsModel, cm::ChoiceMap) = gm_brownian_pos(k, motion, gm)
_dgp(k::Int, gm::GMMaskParams, motion::ISRDynamics, cm::ChoiceMap) = Gen.generate(gm_isr_pos, (k, motion, gm), cm)
_dgp(k::Int, gm::GMMaskParams, motion::ISRPylonsDynamics, cm::ChoiceMap) = Gen.generate(gm_isr_pylons_pos, (k, motion, gm), cm)
_dgp(k::Int, gm::HGMParams, motion::HGMDynamicsModel, cm::ChoiceMap) = Gen.generate(hgm_pos, (k, motion, gm), cm)
_dgp(k::Int, hgm::HGMParams, dm::SquishyDynamicsModel, cm::ChoiceMap) = Gen.generate(squishy_gm_pos, (k, dm, hgm), cm)


function dgp(k::Int, gm::AbstractGMParams,
             motion::AbstractDynamicsModel;
             generate_masks=true,
             cm::ChoiceMap=choicemap())
    
    # new params with all dots having state for data generation
    gm = deepcopy(gm)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    
    # running generative model on just positions (no need to go to masks)
    trace, _ = _dgp(k, gm, motion, cm)
    init_state, states = Gen.get_retval(trace)

    gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
    gt_causal_graphs[1] = init_state.graph
    gt_causal_graphs[2:end] = map(x->x.graph, states)
    
    masks = generate_masks ? get_masks(gt_causal_graphs, gm) : nothing
    
    scene_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :motion => motion,
                       :masks => masks])
end
