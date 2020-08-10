export gm_positions_isr_static

# TODO add (static) to all

@gen function kernel_positions(t::Int,
                                        prev_state::FullState,
                                        dynamics_model::AbstractDynamicsModel,
                                        params::GMMaskParams)

    prev_graph = prev_state.graph

    new_graph = @trace(isr_update(dynamics_model, prev_graph, params), :dynamics)
    new_trackers = new_graph.elements

    new_state = FullState(new_graph, prev_state.record)

    return new_state
end

chain = Gen.Unfold(kernel_positions)

@gen function gm_positions_isr_static(T::Int, motion::AbstractDynamicsModel,
                                           params::GMMaskParams)
    
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(chain(T, init_state, motion, params), :states)

    result = (init_state, states)

    return result
end
