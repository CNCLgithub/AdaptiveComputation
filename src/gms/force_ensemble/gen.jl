export gm_force_ensemble

gp_rfgm = RFGM(MRFS{GaussObs{2}}(), (200, 10.0))

################################################################################
# Initial State
################################################################################

@gen static function fe_tracker_prior(gm::ForceEnsemble)

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


@gen static function fe_ensemble_prior(gm::ForceEnsemble)
    xs, ys = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)
    sigma = @trace(uniform(0.0, xs[2] - xs[1]), :sigma)
    rate = @trace(uniform_discrete(0, gm.n_dots), :rate)
    e::GaussianEnsemble = GaussianEnsemble(SVector{2, Float64}(x, y),
                                           sigma, rate)
    return e
end

@gen static function fe_prior(gm::ForceEnsemble)
    gms = Fill(gm, gm.n_targets)
    trackers = @trace(Gen.Map(fe_tracker_prior)(gms), :tracker_prior)
    ensemble = @trace(fe_ensemble_prior(gm), :ensemble_prior)
    state::ForceEState = ForceEState(gm, trackers, ensemble)
    return state
end

################################################################################
# Dynamics
################################################################################

@gen static function fe_tracker_kernel(gm::ForceEnsemble)
    fx = @trace(normal(0.0, gm.force_sd), :fx)
    fy = @trace(normal(0.0, gm.force_sd), :fy)
    f::SVector{2,Float64} = SVector{2, Float64}(fx, fy)
    return f
end

@gen static function fe_ensemble_kernel(gm::ForceEnsemble, prev::ForceEState)
    dx = @trace(normal(0.0, gm.ens_force_sd), :dx)
    dy = @trace(normal(0.0, gm.ens_force_sd), :dy)
    ds = @trace(normal(0.0, gm.ens_sd), :ds)
    e::GaussianEnsemble = update_ensemble(gm, prev.ensemble, dx, dy, ds)
    return e
end

@gen static function fe_kernel(t::Int64,
                               prev_st::ForceEState,
                               gm::ForceEnsemble)
    # sample some random forces
    gms = Fill(gm, length(prev_st.objects))
    forces = @trace(Gen.Map(fe_tracker_kernel)(gms), :trackers)
    ensemble = @trace(fe_ensemble_kernel(gm, prev_st), :ensemble)
    # integrate noisy forces
    new_state::ForceEState = step(gm, prev_st, forces, ensemble)
    elements = predict(gm, new_state)
    # predict observations as a random finite set
    xs = @trace(gp_rfgm(elements), :masks)
    return new_state
end

################################################################################
# Full models
################################################################################

@gen static function gm_force_ensemble(k::Int, gm::ForceEnsemble)
    init_st = @trace(fe_prior(gm), :init_state)
    states = @trace(Gen.Unfold(fe_kernel)(k, init_st, gm), :kernel)
    result = (init_st, states)
    return result
end
