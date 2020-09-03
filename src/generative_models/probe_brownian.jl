
@gen static function probe_placement(dot::Dot, params::GMMaskParams)
    change = {:probe_flip} ~ bernoulli(params.probe_flip)
    probe = xor(dot.probe, change)
    new_dot = Dot(dot.pos, dot.vel, probe)
    return new_dot
end
map_probe_placement = Gen.Map(probe_placement)

@gen static function probe_kernel(t::Int,
                                 prev_state::FullState,
                                 dynamics_model::AbstractDynamicsModel,
                                 params::GMMaskParams)
    prev_graph = prev_state.graph
    t_graph = @trace(brownian_update(dynamics_model, prev_graph, params), :dynamics)
    pparams = fill(params, params.n_trackers)
    p_trackers = @trace(map_probe_placement(t_graph.elements, pparams), :probes)
    new_graph = update(t_graph, p_trackers)
    pmbrfs = get_masks_params(p_trackers, params)
    @trace(rfs(pmbrfs), :masks)
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end
probe_chain = Gen.Unfold(probe_kernel)

@gen static function probe_brownian(T::Int, motion::AbstractDynamicsModel,
                                params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(probe_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states, nothing)
    return result
end

export probe_brownian
