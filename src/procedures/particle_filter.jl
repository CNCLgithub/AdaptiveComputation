export PopParticleFilter,
    rejuvenate!

using Statistics
using Gen_Compose
using Gen_Compose: initial_args, initial_constraints

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
    rtrace = nothing
    if !isnothing(proc.rejuvenation)
        rtrace = proc.rejuvenation(state, proc.rejuv_args...)
    end
    return rtrace
end

function Gen_Compose.initialize_procedure(proc::PopParticleFilter,
                                          query::SequentialQuery)
    @debug "initializing pf state"
    args = initial_args(query)
    constraints = initial_constraints(query)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           args,
                                           constraints,
                                           proc.particles)
    @debug "initial pf state log weights $(state.log_weights)"
    # @debug "applying initial rejuvination"
    # Gen_Compose.rejuvenate!(proc, state)
    return state
end

function Gen_Compose.smc_step!(state::Gen.ParticleFilterState,
                               proc::PopParticleFilter,
                               query::StaticQuery)


    @debug "smc_step!"
    # Resample before moving on...
    # TODO: Potentially bad for initial step
    Gen_Compose.resample!(proc, state, true)

    # update the state of the particles
    if isnothing(proc.proposal)
        @debug "step without proposal"

        Gen.particle_filter_step!(state, query.args,
                                  (UnknownChange(),),
                                  query.observations)
        @debug "step pf state log weights $(state.log_weights)"
    else
        @debug "step with proposal"
        Gen.particle_filter_step!(state, query.args,
                                  (UnknownChange(),),
                                  query.observations,
                                  proc.proposal,
                                  (query.observations, proc.prop_args...))

    end


    @debug "rejuvinating particles"
    aux_contex = Gen_Compose.rejuvenate!(proc, state)
    
    return aux_contex
end

