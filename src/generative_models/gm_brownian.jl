@gen function brownian_kernel(t::Int,
                              prev_state::State,
                              dynamics_model::BrownianDynamics,
                              params::GMParams)
    new_graph = @trace(brownian_update(dynamics_model, prev_state.graph), :dynamics)
    new_state = State(new_graph)
    return new_state
end

brownian_chain = Gen.Unfold(brownian_kernel)

# generative model for the brownian motion
@gen function gm_brownian(T::Int, motion::BrownianDynamics,
                          params::GMParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(brownian_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end

export GMParams, gm_brownian, default_gm_params
