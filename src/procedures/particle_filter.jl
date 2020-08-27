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
        println("best assignment $(extract_assignments(state.traces[i]))")
    end

    aux_contex = Gen_Compose.rejuvenate!(proc, state)
    
    println()

    return aux_contex
end

