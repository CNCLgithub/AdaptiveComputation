@gen (static) function sample_tracker(cg::CausalGraph)::Dot
    xs, ys, radius = tracker_bounds(cg)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)
    
    ang = @trace(von_mises(0.0, 1e-5), :ang) # super flat
    mag = @trace(normal(vel, 1e-2), :std)

    vx = mag * cos(ang)
    vy = mag * sin(ang)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    return Dot(pos=[x,y,z], vel=[vx, vy], radius=radius)
end

@gen (static) function sample_init_state(cg::CausalGraph)

    ensemble = UniformEnsemble(cg)
    cgs = fill(cg, cg.n_trackers)
    trackers = @trace(Gen.Map(sample_init_tracker)(cgs), :trackers)
    chain_cg = init_cg_from_trackers(cg, ensemble, trackers)
    return chain_cg
end

