export gm_inertia_mask



################################################################################
# Initial State
################################################################################
@gen static function inertia_tracker(gm::AbstractGMParams)::Dot
    xs, ys, radius = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    dm  = get_dm(cg)
    ang = @trace(von_mises(0.0, 1e-5), :ang) # super flat
    mag = @trace(normal(dm.vel, 1e-2), :std)

    vx = mag * cos(ang)
    vy = mag * sin(ang)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    tr = target_rate(gm)
    target = @trace(bernoulli(tr), :target)


    return Dot(pos=[x,y,z], vel=[vx, vy], radius=radius,
               target = target)
end

@gen static function inertia_init(gm::GMParams, dm::InertiaModel,
                                  gr::AbstractGraphics)

    n_trackers = @trace(uniform_discrete(0., gm.max_things),
                        :n_trackers)
    gms = fill(gm, n_trackers)
    trackers = @trace(Gen.Map(inertia_tracker)(gms), :trackers)
    n_tracked_targets = sum(map(target, trackers))

    ens_rate = gm.max_things - n_trackers
    ens_targets = gm.n_targets - n_tracked_targets
    ensemble = UniformEnsemble(gm, gr, ens_rate, ens_targets)

    things = [ensemble; trackers]
    chain_cg = causal_init(gm, dm, gr, things)
    return chain_cg
end


################################################################################
# Dynamics
################################################################################

@gen static function inertial_step(cg::CausalGraph, v::Int64)

    dot = get_prop(cg, v, :object)
    dm = get_dm(cg)

    _x, _y, z = dot.pos
    _vx, _vy = dot.vel

    # transform to angle & magnitude
    ang = atan(_vy, _vx)
    mag = sqrt(_vx^2 + _vy^2)

    # sample inertia
    inertia = @trace(bernoulli(dm.bern), :inertia)

    # sample new angle & magnitude

    #- if high inertia, then flat von_mises
    k = inertia ? dm.k_max : dm.k_min
    ang = @trace(von_mises(ang, k), :ang)

    #- mixture of previous velocity & base
    mu = inertia ? mag : dm.vel
    std = inertia ? dm.w_min : dm.w_max
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

@gen static function inertial_update(prev_cg::CausalGraph, bddiff::Diff)
    # only define diff for trackers that did not die in `bddiff`
    (cgs, vs) = inertia_step_args(prev_cg, bddiff)
    trackers = @trace(Map(inertial_step)(cgs, vs), :trackers)
    tdiff = diff_from_trackers(vs, trackers)
    # merge birth/death and motion diff to be applied in a
    # single step
    diff = merge(bddiff, tdiff)
    return diff
end

@gen static function interial_epistemics(prev_cg::CausalGraph)
    death_rfs = death(prev_cg)
    died = @trace(rfs(death_rfs), :death)

    bl = birth_limit(prev_cg) + died
    to_birth = @trace(uniform_discrete(0., bl), :to_birth)
    bargs = birth_args(prev_cg, died)
    born = @trace(Gen.Map(inertia_tracker)(bargs), :birth)
    diff = birth_diff(prev_cg, born, died)
    return diff
end

@gen static function inertia_kernel(t::Int,
                                    prev_cg::CausalGraph)

    # epistemics kernel - birth/death diff
    bddiff = @trace(inertial_epsitemics(prev_cg), :epistemics)

    # update kinematic state for representations
    # merge with `bdd` for effeciency
    idiff = @trace(inertial_update(prev_cg, bddiff), :dynamics)

    # advancing causal graph (dynamics -> kinematics -> graphics)
    new_cg = causal_update(prev_cg, idiff)

    # predict observations as a random finite set (opt. receptive fields)
    rfs_vec = get_prop(new_cg, :rfs_vec)
    @trace(Map(sample_masks)(rfs_vec), :receptive_fields)
    return new_cg
end

# for position only
@gen static function inertia_pos_kernel(t::Int, prev_cg::CausalGraph)

    # advancing causal graph according to dynamics
    cg = @trace(inertial_update(prev_cg), :dynamics)

    return cg
end

################################################################################
# Full models
################################################################################

@gen static function gm_inertia_mask(k::Int, gm, dm, graphics)
    
    init_cg = @trace(inertia_init(gm, dm, graphics), :init_state)
    states = @trace(Gen.Unfold(inertia_kernel)(k, init_cg), :kernel)
    result = (init_state, states)
    return result
end

# @gen static function gm_inertia_pos(k::Int, gm, dm)

#     cg = get_init_cg(gm, dm)
#     init_state = @trace(inertia_init(cg), :init_state)
#     states = @trace(Gen.Unfold(inertia_pos_kernel)(k, init_state), :kernel)
#     result = (init_state, states)
#     return result
# end
