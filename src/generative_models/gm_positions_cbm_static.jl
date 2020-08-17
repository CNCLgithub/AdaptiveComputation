export gm_positions_cbm_static,
        FullCBMState

struct FullCBMState
    graph::CausalGraph{BDot, SimpleGraph}
    record::RFSElements{Array}
end

# ##### INIT STATE ######
@gen function sample_init_cbm_tracker(params::GMMaskParams)::BDot
    
    init_pos_spread = params.init_pos_spread
    init_vel = params.init_vel

    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)

    z = @trace(uniform(0, 1), :z)
    
    # sampling the angle from the x axis
    bearing = @trace(uniform(-pi, pi), :b)

    # velocity is constant
    vel = init_vel

    # initial velocity is zero
    return BDot([x,y,z], bearing, vel)
end

init_cbm_trackers_map = Gen.Map(sample_init_cbm_tracker)

@gen function sample_init_cbm_state(params::GMMaskParams)
    trackers_params = fill(params, params.n_trackers)
    trackers = @trace(init_cbm_trackers_map(trackers_params), :trackers)
    trackers = collect(BDot, trackers)

    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)
    pmbrfs = RFSElements{Array}(undef, 0)
    return FullCBMState(graph, pmbrfs)
end

@gen (static) function kernel_positions(t::Int,
                                        prev_state::FullCBMState,
                                        dynamics_model::AbstractDynamicsModel,
                                        params::GMMaskParams)

    prev_graph = prev_state.graph

    new_graph = @trace(cbm_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements

    new_state = FullCBMState(new_graph, prev_state.record)

    return new_state
end

chain = Gen.Unfold(kernel_positions)

#@gen (static) function gm_positions_cbm_static(T::Int, motion::AbstractDynamicsModel,
@gen function gm_positions_cbm_static(T::Int, motion::AbstractDynamicsModel,
                                           params::GMMaskParams)
    
    init_state = @trace(sample_init_cbm_state(params), :init_state)
    states = @trace(chain(T, init_state, motion, params), :states)

    result = (init_state, states)

    return result
end
