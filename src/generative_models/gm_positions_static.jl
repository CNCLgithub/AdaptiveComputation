export gm_positions_static

@gen (static) function kernel_positions(t::Int,
                                        prev_state::FullState,
                                        dynamics_model::AbstractDynamicsModel,
                                        params::GMMaskParams)

    prev_graph = prev_state.graph

    new_graph = @trace(brownian_update(dynamics_model, prev_graph), :dynamics)
    new_trackers = new_graph.elements

    new_state = FullState(new_graph, prev_state.pmbrfs_params)

    return new_state
end

chain = Gen.Unfold(kernel_positions)

@gen (static) function gm_positions_static(T::Int, motion::AbstractDynamicsModel,
                                       params::GMMaskParams)
    
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(chain(T, init_state, motion, params), :states)

    result = (init_state, states)

    return result
end
