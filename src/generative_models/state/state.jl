@gen function sample_init_tracker(cg::CausalGraph)::Dot
    @unpack area_width, area_height, dot_radius = (get_gm(cg))
    @unpack vel = (get_dm(cg)) 

    x = @trace(uniform(-area_width/2 + dot_radius, area_width/2 - dot_radius), :x)
    y = @trace(uniform(-area_height/2 + dot_radius, area_height/2 - dot_radius), :y)
    
    ang = @trace(von_mises(0.0, 1e-5), :ang) # super flat
    mag = @trace(normal(vel, 1e-2), :std)

    vx = mag * cos(ang)
    vy = mag * sin(ang)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    return Dot(pos=[x,y,z], vel=[vx, vy], radius=dot_radius)
end

@gen function sample_init_state(cg::CausalGraph)

    @unpack n_trackers = (get_gm(cg))
    cgs = fill(cg, n_trackers)
    init_trackers = @trace(Gen.Map(sample_init_tracker)(cgs), :trackers)
    ensemble = UniformEnsemble(cg)
    
    cg = dynamics_init(cg, [ensemble; init_trackers])
    cg = graphics_init(cg)

    return cg
end

# positional version without ensemble or graphics
@gen function sample_init_state_pos(cg::CausalGraph)

    @unpack n_trackers = (get_gm(cg))

    cgs = fill(cg, n_trackers)
    init_trackers = @trace(Gen.Map(sample_init_tracker)(cgs), :trackers)
    init_trackers = collect(Thing, init_trackers)

    cg = dynamics_init(cg, init_trackers)

    return cg
end
