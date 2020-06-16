export PopParticleFilter,
    rejuvenate!

using Statistics
using Gen_Compose

@with_kw struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = 0.5
    proposal::Union{Gen.GenerativeFunction, Nothing} = nothing
    prop_args::Tuple = Tuple()
    rejuvenation::Union{Function, Nothing} = nothing
    pop_stats::Union{Function, Nothing} = nothing
    stop_rejuv::Union{Function, Nothing} = nothing
    max_sweeps::Int = 1
    max_fails::Int = 1
    verbose::Bool = false
end

function load(Type{PopParticleFilter}, path; kwargs...)
    PopParticleFilter(;read_json(path), kwargs...)
end

mutable struct RejuvTrace
    attempts::Int
    acceptance::Float64
    stats::Any
end

function Gen_Compose.rejuvenate!(proc::PopParticleFilter,
                                 state::Gen.ParticleFilterState)
    
    t, params = get_args(first(state.traces))

    rtrace = RejuvTrace(0, 0, nothing)
    if isnothing(proc.rejuvenation)
        return rtrace
    end
   
    # the only free parameter in this function besides max_sweeps
    rtrace.stats = proc.pop_stats(state)
    sweeps = round(Int, proc.max_sweeps * params["inference_params"]["rejuv_smoothness"]^logsumexp(rtrace.stats))


    println("td confusability: $(rtrace.stats)")
    println("sweeps: $sweeps")
    td_sum = logsumexp(rtrace.stats)
    #println("td sum: $td_sum")

    fails = 0
    # main loop going through rejuvenation
    #for sweep = 1:proc.max_sweeps
    for sweep = 1:sweeps
        #println("sweep $sweep")

        # making a rejuvenation move (rejuvenating velocity)
        rtrace.acceptance += proc.rejuvenation(state, rtrace.stats)
        rtrace.attempts += 1
    
        # computing new population statistics
        new_stats = proc.pop_stats(state)
        
        # early stopping
        if proc.stop_rejuv(new_stats, rtrace.stats)
            fails += 1
            if fails == proc.max_fails break end
        else
            fails = 0
        end

        rtrace.stats = new_stats
    end

    rtrace.acceptance = rtrace.acceptance / rtrace.attempts
    println("acceptance: $(rtrace.acceptance)")
    return rtrace
end

function Gen_Compose.initialize_procedure(proc::ParticleFilter,
                                          query::StaticQuery)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           query.args,
                                           query.observations,
                                           proc.particles)
    rejuvenate!(proc, state)
    return state
end

function Gen_Compose.smc_step!(state::Gen.ParticleFilterState,
                               proc::PopParticleFilter,
                               query::StaticQuery)
    #println("weights: $(Gen.get_log_weights(state))")

    # Resample before moving on...
    # TODO: Potentially bad for initial step
    Gen_Compose.resample!(proc, state, true)

    # update the state of the particles
    if isnothing(proc.proposal)
        Gen.particle_filter_step!(state, query.args,
                                  (UnknownChange(),),
                                  query.observations)
    else
        Gen.particle_filter_step!(state, query.args,
                                  (UnknownChange(),),
                                  query.observations,
                                  proc.proposal,
                                  (query.observations, proc.prop_args...))
    end


    # just getting the MAP TD and A
    t, params = Gen.get_args(first(state.traces))
    println("timestep: $t")
    
    order = sortperm(state.log_weights, rev=true)
    order = [order[1]]
    for i in order
        trace = state.traces[i]
        weight = state.log_weights[i]
        pmbrfs_stats = Gen.get_retval(trace)[2][t].pmbrfs_params.pmbrfs_stats
        td, A, td_weights = pmbrfs_stats.partitions, pmbrfs_stats.assignments, pmbrfs_stats.ll
        #optics = trace[:states => t => :optics]
        masks = trace[:states => t => :masks]
        #td, A, td_weights = get_td_A(pmbrfs, masks, ret.ppp_params, ret.mbrfs_params)
        println("particle weight $weight")
        for j=1:length(td)
            println("TD: $(td[j]) \t A: $(A[j]) \t td weight: $(td_weights[j])")
        end
        println()
    end

    aux_contex = Gen_Compose.rejuvenate!(proc, state)
    println()

    return aux_contex
end

# function Gen_Compose.report_aux!(results::Gen_Compose.InferenceResult,
#                                  aux_state,
#                                  query::Query,
#                                  idx::Int)
#     key = "aux_state/$idx"
#     state_group = Gen_Compose.record_state(results, key, aux_state)
#     return nothing
# end
