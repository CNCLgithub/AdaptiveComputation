export gm_isr

################################################################################
# Initial State
################################################################################
@gen static function isr_dot(gm::ISRGM)
    xs, ys = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(von_mises(0.0, 2*pi), :ang) # super flat
    mag = @trace(normal(gm.vel, 1e-2), :std)

    pos = SVector{2, Float64}([x, y])
    vel = SVector{2, Float64}([mag*cos(ang), mag*sin(ang)])

    target = @trace(bernoulli(gm.target_p), :target)
    new_dot::Dot = Dot(gm, pos, vel, target)
    return new_dot
end

@gen static function isr_init(gm::ISRGM)
    gms = fill(gm, gm.n_dots)
    dots = @trace(Gen.Map(isr_dot)(gms), :init_kernel)
    state::ISRState = ISRState(gm, dots)
    return state
end

################################################################################
# Dynamics
################################################################################

@gen (static) function isr_kernel(t::Int,
                                  prev_st::ISRState,
                                  gm::ISRGM)
    new_dots = step(gm, prev_st)
    next_st::ISRState = ISRState(gm, new_dots)
    return next_st
end


@gen (static) function gm_isr(k::Int, gm::ISRGM)
    init_state = @trace(isr_init(gm), :init_state)
    states = @trace(Gen.Unfold(isr_kernel)(k, init_state, gm), :kernel)
    result = (init_state, states)
    return result
end

