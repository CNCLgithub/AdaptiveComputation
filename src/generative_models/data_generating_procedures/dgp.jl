export dgp

using Setfield

"""
    Helper functions for individual dynamics models.
"""
function _dgp(dm::ISRDynamics, k::Int, gm::GMParams, cm::ChoiceMap)
    Gen.generate(gm_isr_pos, (k, gm, dm), cm)
end
function _dgp(dm::SquishyDynamicsModel, k::Int, gm::GMParams, cm::ChoiceMap)
    Gen.generate(squishy_gm_pos, (k, gm, dm), cm)
end
function _dgp(dm::InertiaModel, k::Int, gm::GMParams, cm::ChoiceMap)
    Gen.generate(gm_inertia_pos, (k, gm, dm), cm)
end


"""
    dgp(k::Int, dm::AbstractDynamicsModel, gm::GMParams;
        cm::ChoiceMap=choicemap())::Vector{CausalGraph}

    Data generating procedure. Returns a Vector{CausalGraph} describing
    the scene for each time step.
...
# Arguments:
- k: number of time steps
- dm: dynamics model parameters
- gm: generative model parameters
...
# Optional arguments
- cm: choicemap with constraints
"""
function dgp(k::Int, dm::AbstractDynamicsModel, gm::GMParams;
             cm::ChoiceMap=choicemap())::Vector{CausalGraph}
    
    # new params with all dots having state for data generation
    gm = deepcopy(gm)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    
    trace, _ = _dgp(dm, k, gm, cm)
    init_cg, cgs = Gen.get_retval(trace)

    gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
    gt_causal_graphs[1] = init_cg
    gt_causal_graphs[2:end] = cgs
    
    return gt_causal_graphs
end
