export gm_inertia_mask



################################################################################
# Initial State
################################################################################
@gen static function inertia_tracker(gm::GMParams,
                                     dm::InertiaModel)::Dot
    xs, ys, radius = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(von_mises(0.0, 1e-5), :ang) # super flat
    mag = @trace(normal(dm.vel, 1e-2), :std)

    vx = mag * cos(ang)
    vy = mag * sin(ang)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    target = @trace(bernoulli(gm.target_rate), :target)

    return Dot(pos=[x,y,z], vel=[vx, vy],
               radius=radius,
               target = target)
end

@gen static function inertia_init(gm::GMParams, dm::InertiaModel,
                                  gr::AbstractGraphics)

    n_trackers = @trace(uniform_discrete(0, gm.max_things),
                        :n_trackers)
    gms = fill(gm, n_trackers)
    dms = fill(dm, n_trackers)
    trackers = @trace(Gen.Map(inertia_tracker)(gms, dms), :trackers)
    n_tracked_targets = n_trackers == 0 ? 0 : sum(map(target, trackers))

    ens_rate = gm.max_things - n_trackers
    ens_targets = gm.n_targets - n_tracked_targets
    ensemble = UniformEnsemble(gm, gr, ens_rate, ens_targets)

    things = collect(Thing, [ensemble; trackers])
    chain_cg = causal_init(gm, dm, gr, things)
    init_st = InertiaKernelState(chain_cg,
                                 RFSElements{Any}(undef, 0),
                                 [],
                                 falses(0,0,0),
                                 Float64[])
    return init_st
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

    # sampl new angle & magnitude

    #- if high inertia, then flat von_mises
    k = inertia ? dm.k_max : dm.k_min
    ang_mu = !inertia * pi # approximate collisions
    ang = @trace(von_mises(ang + ang_mu, k), :ang)

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

    d = Dot(pos = [x,y,z], vel = [vx, vy],
            radius = dot.radius,
            target = dot.target)
    return d
end

@gen static function inertial_update(prev_cg::CausalGraph, bddiff::Diff)
    # only define diff for trackers that did not die in `bddiff`
    (cgs, vs) = inertia_step_args(prev_cg, bddiff)
    trackers = @trace(Map(inertial_step)(cgs, vs), :trackers)
    tdiff = diff_from_trackers(vs, collect(Dot, trackers))
    # merge birth/death and motion diff to be applied in a
    # single step
    diff = merge(bddiff, tdiff)
    return diff
end

@gen static function inertial_epistemics(prev_cg::CausalGraph)
    dm = get_dm(prev_cg)
    death_rfs = death(dm, prev_cg)
    died = @trace(rfs(death_rfs), :death)
    ndied = length(died)

    bl = birth_limit(dm, prev_cg, ndied)
    # to_birth = @trace(uniform_discrete(0, bl), :to_birth)
    to_birth = @trace(bernoulli(bl), :to_birth)
    (gms, dms) = birth_args(dm, prev_cg, to_birth)
    born = @trace(Gen.Map(inertia_tracker)(gms, dms), :birth)
    diff = birth_diff(dm, prev_cg,
                      collect(Thing, born),
                      collect(Int64, died))
    return diff
end

@gen static function inertia_kernel(t::Int64,
                             prev_st::InertiaKernelState)
# @gen function inertia_kernel(t::Int64,

    prev_cg = prev_st.world

    # epistemics kernel - birth/death diff
    bddiff = @trace(inertial_epistemics(prev_cg), :epistemics)

    # update kinematic state for representations
    # merge with `bdd` for effeciency
    idiff = @trace(inertial_update(prev_cg, bddiff), :dynamics)

    # advancing causal graph (dynamics -> kinematics -> graphics)
    new_cg = causal_update(get_dm(prev_cg), prev_cg, idiff)

    # predict observations as a random finite set (opt. receptive fields)
    es = get_prop(new_cg, :es)
    xs = @trace(rfs(es), :masks)

    # store the associations for later use
    current_state = InertiaKernelState(new_cg,
                                       es,
                                       xs)

    return current_state
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
    result = (init_cg, states)
    return result
end
