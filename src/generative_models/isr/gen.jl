export gm_isr_pos

################################################################################
# Initial State
################################################################################
@gen static function isr_tracker(cg::CausalGraph)::Dot
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

@gen static function isr_init(cg::CausalGraph)
    cgs = fill(cg, cg.n_trackers)
    trackers = @trace(Gen.Map(isr_tracker)(cgs), :trackers)
    chain_cg = init_cg_from_trackers(cg, trackers)
    return chain_cg
end

################################################################################
# Dynamics
################################################################################

@gen function isr_step(cg::CausalGraph, v::Int64)::Dot
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
    cg = dynamics_update(get_dm(cg), cg, things)

    # then brownian step (random)
    cgs = fill(cg, length(vs))
    things = @trace(Map(isr_step)(cgs, vs), :trackers)
    cg = dynamics_update(get_dm(cg), cg, things)

    return cg
end


@gen function isr_pos_kernel(t::Int,
                         prev_cg::CausalGraph)
    # advancing causal graph according to dynamics
    # (there is a deepcopy here)
    cg = @trace(isr_update(prev_cg), :dynamics)
    return cg
end


@gen function gm_isr_pos(k::Int, gm, dm)
    cg = get_init_cg(gm, dm)
    init_state = @trace(isr_init(cg), :init_state)
    states = @trace(Gen.Unfold(isr_pos_kernel)(k, init_state), :kernel)
    result = (init_state, states)
    return result
end

