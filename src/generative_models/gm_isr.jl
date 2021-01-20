#@gen static function isr_kernel(t::Int,
@gen function isr_kernel(t::Int,
                                prev_state::State,
                                dynamics_model::ISRDynamics,
                                params::GMParams)
    new_graph = @trace(isr_update(dynamics_model, prev_state.graph, params), :dynamics)
    new_state = State(new_graph)
    return new_state
end

isr_chain = Gen.Unfold(isr_kernel)

# generative model for ISR motion
#@gen static function gm_isr(T::Int, motion::ISRDynamics,
@gen function gm_isr(T::Int, motion::ISRDynamics,
                                 params::GMParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(isr_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end

export gm_isr
