
@gen (static) function inertial_step(cg::CausalGraph, v::Int64)

    dot = get_prop(cg, v, :object)
    dm = get_dm(cg)

    _x, _y, z = dot.pos
    _vx, _vy = dot.vel

    # transform to angle & magnitude
    ang = atan(_vy, _vx)
    mag = sqrt(_vx^2 + _vy^2)
    
    # sample inertia
    # inertia = @trace(beta(dm.a, dm.b), :inertia)
    inertia = @trace(bernoulli(dm.bern), :inertia)

    # sample new angle & magnitude

    #- if high inertia, then flat von_mises
    k = max(dm.k_min, inertia * dm.k_max) # fixing bug in vonmises with small k
    ang = @trace(von_mises(ang, k), :ang)

    #- mixture of previous velocity & base
    mu = inertia * mag + (1.0 - inertia) * dm.vel
    std = max(dm.w_min, (1.0 - inertia) * dm.w_max)
    mag = @trace(normal(mu, std), :mag)

    # converting back to vector form
    vx = mag * cos(ang)
    vy = mag * sin(ang)

    x = _x + vx
    y = _y + vy
    z = @trace(uniform(0, 1), :z)

    d = Dot(pos = [x,y,z], vel = [vx, vy], radius = dot.radius)
    return d
end

@gen (static) function inertial_update(prev_cg::CausalGraph)
    (cgs, vs) = inertia_step_args(prev_cg)
    things = @trace(Map(inertial_step)(cgs, vs), :brownian)
    new_cg = dynamics_update(get_dm(prev_cg), prev_cg, things)
    return new_cg
end

