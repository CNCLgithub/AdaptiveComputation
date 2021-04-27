export dgp

using Setfield

_dgp(k::Int, gm::GMParams, dm::AbstractDynamicsModel, cm::ChoiceMap) = error("not implemented")
_dgp(k::Int, gm::GMParams, dm::BrownianDynamicsModel, cm::ChoiceMap) = Gen.generate(gm_brownian_pos, (k, dm, gm), cm)
_dgp(k::Int, gm::GMParams, dm::ISRDynamics, cm::ChoiceMap) = Gen.generate(gm_isr_pos, (k, dm, gm), cm)
_dgp(k::Int, gm::GMParams, dm::ISRPylonsDynamics, cm::ChoiceMap) = Gen.generate(gm_isr_pylons_pos, (k, dm, gm), cm)
_dgp(k::Int, gm::HGMParams, dm::HGMDynamicsModel, cm::ChoiceMap) = Gen.generate(hgm_pos, (k, dm, gm), cm)
_dgp(k::Int, hgm::HGMParams, dm::SquishyDynamicsModel, cm::ChoiceMap) = Gen.generate(squishy_gm_pos, (k, dm, hgm), cm)


function dgp(k::Int, gm::AbstractGMParams,
             dm::AbstractDynamicsModel;
             generate_masks=true,
             generate_cm=false,
             cm::ChoiceMap=choicemap())
    
    # new params with all dots having state for data generation
    gm = deepcopy(gm)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    
    # running generative model on just positions (no need to go to masks)
    trace, _ = _dgp(k, gm, dm, cm)
    init_state, states = Gen.get_retval(trace)

    gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
    gt_causal_graphs[1] = init_state.cg
    gt_causal_graphs[2:end] = map(x->x.cg, states)
    
    masks = generate_masks ? get_masks(gt_causal_graphs, gm) : nothing
    cm = generate_cm ? Gen.get_choices(trace) : nothing
    
    scene_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :dm => dm,
                       :masks => masks,
                       :cm => cm])
end
