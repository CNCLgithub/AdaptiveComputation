export dgp

using Setfield

_dgp(k::Int, gm::GMMaskParams, motion::AbstractDynamicsModel) = error("not implemented")
_dgp(k::Int, gm::GMMaskParams, motion::BrownianDynamicsModel) = gm_brownian_pos(k, motion, gm)
_dgp(k::Int, gm::GMMaskParams, motion::ISRDynamics) = gm_isr_pos(k, motion, gm)

_dgp(k::Int, gm::GMMaskParams, motion::ISRPylonsDynamics) = gm_isr_pylons_pos(k, motion, gm)
_dgp(k::Int, gm::GMMaskParams, motion::ISRPylonsDynamics, cm::ChoiceMap) = Gen.generate(gm_isr_pylons_pos, (k, motion, gm), cm)


function dgp(k::Int, gm::GMMaskParams,
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
    
    trial_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :motion => motion,
                       :masks => masks])
end

# function dgp(k::Int, gm::GMMaskParams,
             # motion::AbstractDynamicsModel;
             # generate_masks=true)

    # # new params with all dots having state for data generation
    # gm = deepcopy(gm)
    # gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    
    # # running generative model on just positions (no need to go to masks)
    # init_state, states = _dgp(k, gm, motion)
    
    # gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
    # gt_causal_graphs[1] = init_state.graph
    # gt_causal_graphs[2:end] = map(x->x.graph, states)
    
    # masks = generate_masks ? get_masks(gt_causal_graphs, gm) : nothing
    
    # trial_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       # :motion => motion,
                       # :masks => masks])
# end
