export AbstractAttentionModel,
    get_stats,
    get_sweeps,
    early_stopping,
    rejuvenate_attention!

abstract type AbstractAttentionModel end

function get_stats(::AbstractAttentionModel, ::Gen.ParticleFilterState)
    error("not implemented")
end

function get_sweeps(::AbstractAttentionModel, stats)
    error("not implemented")
end

function early_stopping(::AbstractAttentionModel, prev_stats, new_stats)
    error("not implemented")
end

function rejuvenate_attention!(pf_state::Gen.ParticleFilterState, attention::AbstractAttentionModel)



    t, motion, gm = get_args(first(pf_state.traces))

    rtrace = RejuvTrace(0, 0, nothing, zeros(gm.n_trackers))

    # CHANGE (added rejuvenation for all before applying attention)
    n_rejuvs = 3
    for i = 1:n_rejuvs
        for j = 1:gm.n_trackers
            weights = zeros(gm.n_trackers)
            weights[j] = 1.0
            acceptance, attended_trackers = perturb_state!(pf_state, weights)
            # rtrace.attended_trackers += attended_trackers
        end
    end

    rtrace.stats = get_stats(attention, pf_state)
    rtrace.attended_trackers = rtrace.stats # CHANGED this reflects the actual underlying continuous
    #weights = sum(rtrace.stats) == 0 ? fill(1.0/gm.n_trackers, gm.n_trackers) : softmax(rtrace.stats)
    weights = sum(rtrace.stats) == 0 ? fill(1.0/gm.n_trackers, gm.n_trackers) : rtrace.stats/sum(rtrace.stats)
    sweeps = get_sweeps(attention, rtrace.stats)

    
    println("categorical weights: $weights")
    println("sweeps: $sweeps")

    fails = 0
    # main loop going through rejuvenation
    for sweep = 1:sweeps

        # making a rejuvenation move (rejuvenating velocity)
        #
        acceptance, attended_trackers = perturb_state!(pf_state, weights)
        rtrace.acceptance += acceptance
        # rtrace.attended_trackers += attended_trackers
        rtrace.attempts += 1

        # computing new population statistics
        # new_stats = get_stats(attention, pf_state)

        # # early stopping
        # if early_stopping(attention, new_stats, rtrace.stats)
        #     break
        # end

        # rtrace.stats = new_stats
    end

    rtrace.acceptance = rtrace.acceptance / rtrace.attempts
    println("acceptance: $(rtrace.acceptance)")
    println("attended_trackers: $(rtrace.attended_trackers)")
    return rtrace
end


include("perturb_state.jl")

include("td_entropy.jl")
include("uniform.jl")
include("sensitivity.jl")
