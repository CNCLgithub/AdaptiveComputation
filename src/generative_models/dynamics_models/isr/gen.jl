@gen function isr_brownian_step(cg::CausalGraph, v::Int64)::Dot
    dm = get_dm(cg)
    dot = get_prop(cg, v, :object)

    _x, _y, _z = dot.pos
    vx, vy = dot.vel
    
    if dm.brownian
        vx = @trace(normal(dm.inertia * vx - dm.spring * _x,
                               dm.sigma_x), :vx)
        vy = @trace(normal(dm.inertia * vy - dm.spring * _y,
                               dm.sigma_y), :vy)
    end

    x = _x + vx
    y = _y + vy
    
    return Dot(pos=[x,y,_z], vel=[vx,vy])
end



@gen function isr_update(prev_cg::CausalGraph)
    cg = deepcopy(prev_cg)
    vs = get_object_verts(cg, Dot)

    # first start with repulsion step (deterministic)
    things = isr_repulsion_step(cg)
    dynamics_update!(get_dm(cg), cg, things)

    # then brownian step (random)
    cgs = fill(cg, length(vs))
    things = @trace(Map(isr_brownian_step)(cgs, vs), :brownian)
    dynamics_update!(get_dm(cg), cg, things)

    return cg
end

