export PopParticleFilter,
    rejuvenate!

using GenParticleFilters
using Gen_Compose
using Gen_Compose: initial_args, initial_constraints,
    AuxillaryState, PFChain

@with_kw struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    attention::AbstractAttentionModel
end

# function load(::Type{PopParticleFilter}, path; kwargs...)
#     PopParticleFilter(;read_json(path)..., kwargs...)
# end

function Gen_Compose.PFChain{Q, P}(q::Q,
                                   p::P,
                                   n::Int,
                                   i::Int = 1) where
    {Q<:SequentialQuery,
     P<:PopParticleFilter}

    state = Gen_Compose.initialize_procedure(p, q)
    aux = AdaptiveComputation(p.attention)
    return PFChain{Q, P}(q, p, state, aux, i, n)
end

function Gen_Compose.step!(chain::PFChain{<:SequentialQuery, <:PopParticleFilter})
    @unpack query, proc, state, step = chain
    @show step
    squery = query[step]
    @unpack args, argdiffs, observations = squery
    # Resample before moving on...
    # Gen.maybe_resample!(state, ess_threshold=proc.ess)
    if effective_sample_size(state) < proc.ess
        # Perform residual resampling, pruning low-weight particles
        pf_residual_resample!(state)
    end
    # update the state of the particles
    Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    @unpack attention = proc
    adaptive_compute!(chain, attention)
    return nothing
end
