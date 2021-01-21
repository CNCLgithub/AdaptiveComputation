struct State
    graph::CausalGraph{Dot, SimpleGraph}
end

@gen function sample_init_tracker(init_pos_spread::Real)::Dot
    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)
    vx = 0.0
    vy = 0.0

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    # initial velocity is zero
    return Dot([x,y,z], [vx, vy])
end

init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(gm::GMParams)
    init_pos_spread_vec = fill(gm.init_pos_spread, gm.n_trackers)
    trackers = @trace(init_trackers_map(init_pos_spread_vec), :trackers)
    trackers = collect(Dot, trackers)

    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)

    State(graph)
end
