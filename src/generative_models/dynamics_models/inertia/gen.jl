
@gen function inertial_step(dm::InertiaModel, dot::Dot)
    _x, _y, z = dot.pos
    _vx, _vy = dot.vel
    _ax, _ay = dot.acc

    acc = @trace(beta(dm.a, dm.b), :acc)

    vel_sd = min(dm.low_w/acc, dm.high_w)
    vx = @trace(normal(acc * _vx, vel_sd), :vx)
    vy = @trace(normal(acc * _vy, vel_sd), :vy)

    vel = [vx, vy] .+ 1e-10
    vel *= dm.vel/norm(vel)
    vel = @trace(broadcasted_normal(vel, dm.low_w), :v)

    # vel_sd = min(dm.low_w/acc, dm.high_w)
    # vx = @trace(normal(acc * _vx, vel_sd), :vx)
    # vy = @trace(normal(acc * _vy, vel_sd), :vy)
    # vel = [vx, vy]

    x = _x + vel[1]
    y = _y + vel[2]
    z = @trace(uniform(0, 1), :z)

    d = Dot(pos = [x,y,z], vel = vel, acc = [acc, acc])
    return d
end


@gen function inertial_update(dm::InertiaModel, cg::CausalGraph)
    dots = get_objects(cg, Dot)
    temp_state = @trace(Map(inertial_step)(fill(dm, length(dots)), dots), :brownian)
    new_cg = process_temp_state(temp_state, cg, dm)
    return new_cg
end

export InertiaModel
