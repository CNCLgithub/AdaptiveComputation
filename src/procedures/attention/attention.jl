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
    @debug "attention"
    args = get_args(first(pf_state.traces))
    t, gm, dm, graphics = args
    
    rtrace = RejuvTrace(0, 0, nothing, zeros(gm.n_trackers))
    
    @debug "log weights: $(pf_state.log_weights)"

    @time rtrace.stats = get_stats(attention, pf_state)
    weights = sum(rtrace.stats) == 0 ? fill(1.0/gm.n_trackers, gm.n_trackers) : get_weights(attention, rtrace.stats)
    sweeps = get_sweeps(attention, rtrace.stats)

    @debug "attention weights $(weights)"
    @debug "compute cycles $(sweeps)"

    # main loop going through rejuvenation
    @debug "applying compute cycles"
    # Profile.init(delay = 1E-4,
    #              n = 10^6)
    @time (acceptance, attended_trackers) = perturb_state!(pf_state, attention,
                                                     weights, sweeps)
    rtrace.acceptance += acceptance
    rtrace.attended_trackers += attended_trackers
    rtrace.attempts += sweeps

    # rtrace.acceptance = rtrace.acceptance / rtrace.attempts
    @debug "acceptance: $(rtrace.acceptance)"
    @debug "attended_trackers: $(rtrace.attended_trackers)"
    # just getting the MAP TD and A
    t, gm = Gen.get_args(first(pf_state.traces))
    @debug "timestep: $t"

    return rtrace
end

include("uniform.jl")
include("sensitivity.jl")

include("perturb_state.jl")

