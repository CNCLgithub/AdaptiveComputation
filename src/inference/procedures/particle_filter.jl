export PopParticleFilter,
    rejuvenate!

using Gen_Compose
using Gen_Compose: initial_args, initial_constraints,
    AuxillaryState, SeqPFChain

@with_kw struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    attention::AbstractAttentionModel
end

function load(::Type{PopParticleFilter}, path; kwargs...)
    PopParticleFilter(;read_json(path)..., kwargs...)
end

function Gen_Compose.initialize_chain(proc::PopParticleFilter,
                                      query::SequentialQuery)
    @debug "initializing pf state"
    args = initial_args(query)
    constraints = initial_constraints(query)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           args,
                                           constraints,
                                           proc.particles)

    aux = AdaptiveComputation(proc.attention)
    return SeqPFChain(query, proc, state, aux)
end

function Gen_Compose.smc_step!(chain::SeqPFChain, proc::PopParticleFilter,
                               query::StaticQuery)
    @unpack state = chain
    @unpack args, observations = query
    # Resample before moving on...
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    # update the state of the particles
    argdiffs = (UnknownChange(), NoChange())
    println("taking step $(first(args))")
    @time Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    @unpack attention = proc
    adaptive_compute!(chain, attention)
    return nothing
end
