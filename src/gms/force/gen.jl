export gm_force

mo_rfs = MRFS{MotionObs}()
mo_rfgm = RFGM(MRFS{MotionObs}(), (200, 2.0))

################################################################################
# Initial State
################################################################################

@gen static function force_tracker(gm::ForceGM)

    xs, ys = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(uniform(0., 2*pi), :ang)
    mag = @trace(normal(gm.vel, 1e-2), :std)

    pos = SVector{2, Float64}(x, y)
    vel = SVector{2, Float64}(mag*cos(ang), mag*sin(ang))

    target = @trace(bernoulli(gm.target_p), :target)
    new_dot::Dot = Dot(gm, pos, vel, target)
    return new_dot
end

@gen static function force_init(gm::ForceGM)
    gms = Fill(gm, gm.n_dots)
    trackers = @trace(Gen.Map(force_tracker)(gms), :init_kernel)
    state::ForceState = ForceState(gm, trackers)
    return state
end

################################################################################
# Dynamics
################################################################################

@gen static function force_prior(gm::ForceGM)
    fx = @trace(normal(0.0, gm.force_sd), :fx)
    fy = @trace(normal(0.0, gm.force_sd), :fy)
    f::SVector{2,Float64} = SVector{2, Float64}(fx, fy)
    return f
end

@gen static function force_kernel(t::Int64,
                                    prev_st::ForceState,
                                    gm::ForceGM)
    # sample some random forces
    gms = Fill(gm, gm.n_dots)
    forces = @trace(Gen.Map(force_prior)(gms), :trackers)
    # integrate noisy forces
    new_state::ForceState = step(gm, prev_st, forces)
    elements = predict(gm, new_state)
    # predict observations as a random finite set
    xs = @trace(mo_rfgm(elements), :masks)
    return new_state
end

################################################################################
# Full models
################################################################################

@gen static function gm_force(k::Int, gm::ForceGM)
    init_st = @trace(force_init(gm), :init_state)
    states = @trace(Gen.Unfold(force_kernel)(k, init_st, gm), :kernel)
    result = (init_st, states)
    return result
end
