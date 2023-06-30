export query_from_params

"Package simulated trajecties into a vector of choicemaps"
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

"Creates a `choicemap` for t0 (gt initial positions)"
function get_init_constraints(gm::InertiaGM, st::InertiaState)
    cm = Gen.choicemap()
    n = gm.n_targets
    for i=1:n
        pos = get_pos(st.objects[i])
        addr = :init_state => :init_kernel => i => :x
        cm[addr] = pos[1]
        addr = :init_state => :init_kernel => i => :y
        cm[addr] = pos[2]
        # by convention, the first n init_kernel are targets
        # in the source trace
        cm[:init_state => :init_kernel => i => :target] = true
    end
    return Gen.StaticChoiceMap(cm)
end


"""
    query_from_params(gm, gt_state)

The `SequentialQuery` for an MOT trial.

Extracts trajectories from a vector of states (`gm_states`)
and uses the parameters of `gm` to generate:
 - observations (masks)
 - initial constraints for simulation
"""
function query_from_params(gm::InertiaGM,
                           gt_states::Vector{InertiaState})

    k = length(gt_states)
    init_gt = gt_states[1]
    rest_gt = gt_states[2:end]

    # initialize with gt positions for t = 1
    init_constraints = get_init_constraints(gm, init_gt)
    # must infer trajectories for t > 1
    observations = get_observations(gm, rest_gt)

    init_args = (0, gm)
    args = [(t, gm) for t in 1:k]
    argdiffs = [(UnknownChange(), NoChange()) for _ = 1:k]

    latent_map = LatentMap(
        :auxillary => digest_auxillary,
        :positions => digest_tracker_positions,
        :task_accuracy => digest_td_accuracy,
    )

    Gen_Compose.SequentialQuery(latent_map,
                                gm_inertia,
                                init_args,
                                init_constraints,
                                args,
                                argdiffs,
                                observations)
end


function chain_performance(dg)
    # causal graphs at the end of inference
    perf = dg[end, :task_accuracy]
    df = DataFrame(tracker = 1:length(perf),
                   td_acc = perf)
    return df
end

function chain_attention(dg, n_targets = 4)
    aux_state = dg[:, :auxillary]

    steps = length(aux_state)
    cycles = 0

    df = DataFrame(
        frame = Int64[],
        tracker = Int64[],
        importance = Float64[],
        cycles = Float64[],
        sensitivity = Float64[],
        pred_x = Float64[],
        pred_y = Float64[],
        pred_x_sd = Float64[],
        pred_y_sd = Float64[])
    for frame = 1:steps
        arrousal = aux_state[frame].arrousal
        cycles += arrousal
        importance = aux_state[frame].importance
        sensitivity = aux_state[frame].sensitivities
        cycles_per_latent =  arrousal .* importance
        avg_pos = dg[frame, :positions].avg
        sd_pos = dg[frame, :positions].sd
        for i = 1:n_targets
            px, py = avg_pos[1, i, :]
            sx, sy = sd_pos[1, i, :]
            push!(df, (frame, i,
                       importance[i],
                       cycles_per_latent[i],
                       sensitivity[i],
                       px, py, sx, sy))
        end
    end
    println("total arrousal = $(cycles)")
    return df
end
