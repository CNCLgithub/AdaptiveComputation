@gen static function isr_kernel(t::Int,
                                prev_state::State,
                                dynamics_model::ISRDynamics,
                                params::GMParams)
    new_graph = @trace(isr_update(dynamics_model, prev_state.graph), :dynamics)
    new_state = State(new_graph)
    return new_state
end

brownian_chain = Gen.Unfold(brownian_kernel)

# generative model for the brownian motion
@gen static function gm_brownian(k::Int, motion::ISRDynamics,
                                 params::GMParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(brownian_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end

export GMParams, gm_brownian, default_gm_params
