export query_from_params

function get_observations(gm::InertiaGM,
                          st::Vector{InertiaState})
    k = length(st)
    observations = Vector{Gen.ChoiceMap}(undef, k)
    for t=1:k
        cm = Gen.choicemap()
        cm[:kernel => t => :masks] = st[t].xs
        observations[t] = Gen.StaticChoiceMap(cm)
    end
    return observations
end

function get_init_constraints(gm::InertiaGM, st::InertiaState)
    cm = Gen.choicemap()
    n = gm.n_targets
    for i=1:n
        dot = st.objects[i]
        addr = :init_state => :init_kernel => i => :x
        cm[addr] = dot.pos[1]
        addr = :init_state => :init_kernel => i => :y
        cm[addr] = dot.pos[2]
        # by convention, the first n init_kernel are targets
        # in the source trace
        cm[:init_state => :init_kernel => i => :target] = true
    end
    return Gen.StaticChoiceMap(cm)
end


function query_from_params(gm::InertiaGM,
                           gt_states::Vector{InertiaState},
                           k::Int64)

    # ensure that all obs are present
    dgp_gm = setproperties(gm,
                           (outer_f = 1.0,
                            inner_f = 1.0))

    init_gt = gt_states[1]
    rest_gt = gt_states[2:end]

    init_constraints = get_init_constraints(dgp_gm,
                                            init_gt)
    observations = get_observations(dgp_gm, rest_gt)

    init_args = (0, gm)
    args = [(t, gm) for t in 1:k]

    latent_map = LatentMap(
        :auxillary => digest_auxillary,
        :positions => digest_tracker_positions
    )

    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_inertia,
                                        init_args,
                                        init_constraints,
                                        args,
                                        observations)

    # q = first(query)
    # ms = q.observations[:kernel => 1 => :masks]
    # @debug "number of masks $(length(ms))"
    return query
end
