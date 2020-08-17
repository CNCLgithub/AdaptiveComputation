export PopParticleFilter,
    rejuvenate!

using Statistics
using Gen_Compose

@with_kw struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles/2.0
    proposal::Union{Gen.GenerativeFunction, Nothing} = nothing
    prop_args::Tuple = ()
    rejuvenation::Union{Function, Nothing} = nothing
    rejuv_args::Tuple = ()
end

function load(::Type{PopParticleFilter}, path; kwargs...)
    PopParticleFilter(;read_json(path)..., kwargs...)
end

mutable struct RejuvTrace
    attempts::Int
    acceptance::Float64
    stats::Any
    attended_trackers::Vector{Float64}
end

function Gen_Compose.rejuvenate!(proc::PopParticleFilter,
                                 state::Gen.ParticleFilterState)
    rtrace = proc.rejuvenation(state, proc.rejuv_args...)
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
    t, gm = Gen.get_args(first(state.traces))
    println("timestep: $t")

    order = sortperm(state.log_weights, rev=true)
    order = [order[1]]
    for i in order
        trace = state.traces[i]
        weight = state.log_weights[i]

        # pmbrfs_stats = Gen.get_retval(trace)[2][t].pmbrfs_params.pmbrfs_stats
        # td, td_weights = pmbrfs_stats.partitions, pmbrfs_stats.ll_partitions
        # A, A_weights = pmbrfs_stats.assignments, pmbrfs_stats.ll_assignments

        # println("particle weight $weight")
        # for j=1:length(td)
        #     println("TD: $(td[j][1]) \t A: $(td[j][2]) \t td weight: $(td_weights[j])")
        # end
        # println()
    end

    aux_contex = Gen_Compose.rejuvenate!(proc, state)
    
    println()

    return aux_contex
end

