export gm_inertia



################################################################################
# Initial State
################################################################################
@gen static function inertia_tracker(gm::InertiaGM)

    xs, ys = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(uniform(0., 2*pi), :ang)
    mag = @trace(normal(gm.vel, 1e-2), :std)

    pos = SVector{2, Float64}([x, y])
    vel = SVector{2, Float64}([mag*cos(ang), mag*sin(ang)])

    target = @trace(bernoulli(gm.target_p), :target)
    new_dot::Dot = Dot(gm, pos, vel, target)
    return new_dot
end


@gen static function inertia_init(gm::InertiaGM)
    gms = fill(gm, gm.n_targets)
    trackers = @trace(Gen.Map(inertia_tracker)(gms), :init_kernel)
    state::InertiaState = InertiaState(gm, trackers)
    return state
end


################################################################################
# Dynamics
################################################################################

@gen static function inertia_step(gm::InertiaGM, d::Dot)

    _x, _y = d.pos
    _vx, _vy = d.vel

    # transform to angle & magnitude
    ang_mu = atan(_vy, _vx)
    mag = sqrt(_vx^2 + _vy^2)

    # sample inertia
    inertia = @trace(bernoulli(gm.bern), :inertia)

    # sampl new angle & magnitude

    #- if high inertia, then turn 180 deg
    ang_turn = !inertia * pi # approximate collisions
    ang = @trace(von_mises(ang_mu, gm.k), :ang) + ang_turn

    #- mixture of previous velocity & base
    # mag = @trace(normal(mag, gm.w), :mag)
    mag = @trace(normal(gm.vel, gm.w), :mag)

    # converting back to vector form
    vel = SVector{2, Float64}([mag * cos(ang), mag * sin(ang)])
    # vx = mag * cos(ang)
    # vy = mag * sin(ang)

    pos = d.pos + vel # SVector{2, Float64}([_x + vx, _y + vy])

    ku::KinematicsUpdate = KinematicsUpdate(pos, vel)
    return ku
end

@gen static function inertia_kernel(t::Int64,
                                    prev_st::InertiaState,
                                    gm::InertiaGM)


    # update kinematic state for representations
    (gms, dots) = inertia_step_args(gm, prev_st)
    kupdates = @trace(Gen.Map(inertia_step)(gms, dots), :trackers)

    # advancing causal graph (dynamics -> kinematics -> graphics)
    new_dots = step(gm, prev_st, kupdates)

    # predict observations as a random finite set
    es = predict(gm, prev_st, new_dots)
    # xs = @trace(point_mrfs(es, 20, 1.0), :masks)
    xs = @trace(mask_mrfs(es, 20, 1.0), :masks)
    # xs = @trace(mask_rfs(es), :masks)

    # store the associations for later use
    current_state::InertiaState = InertiaState(prev_st,
                                 new_dots,
                                 es,
                                 xs)
    return current_state
end

################################################################################
# Full models
################################################################################

@gen static function gm_inertia(k::Int, gm::InertiaGM)
    init_st = @trace(inertia_init(gm), :init_state)
    states = @trace(Gen.Unfold(inertia_kernel)(k, init_st, gm), :kernel)
    result = (init_st, states)
    # result::Tuple{InertiaState, Vector{InertiaState}} = (init_st, states)
    return result
end
