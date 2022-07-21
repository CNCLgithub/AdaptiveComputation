export PopParticleFilter,
    rejuvenate!

using Statistics
using Gen_Compose
using Gen_Compose: initial_args, initial_constraints,
    AuxillaryState, SeqPFChain

@with_kw struct PopParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    proposal::Union{Gen.GenerativeFunction, Nothing} = nothing
    prop_args::Tuple = ()
    rejuvenation::Union{Function, Nothing} = nothing
    rejuv_args::Tuple = ()
end

function load(::Type{PopParticleFilter}, path; kwargs...)
    PopParticleFilter(;read_json(path)..., kwargs...)
end

@with_kw mutable struct AdaptiveComputation <: AuxillaryState
    attempts::Int64 = 0
    acceptance::Float64 = 0.
    cycles::Vector{Int64} = Int64[]
    arrousal::Vector{Float64} = Float64[]
    weights::Dict{Int64, Vector{Float64}} = Dict{Int64, Vector{Float64}}()
    sensitivities::Dict{Int64, Vector{Float64}} = Dict{Int64, Vector{Float64}}()
    allocated::Dict{Int64, Vector{Int64}} = Dict{Int64, Vector{Int64}}()
end

function Gen_Compose.rejuvenate!(chain::SeqPFChain,
                                 proc::PopParticleFilter)
    @unpack rejuvenation, rejuv_args = proc
    if !isnothing(rejuvenation)
        rejuvenation(chain, rejuv_args...)
    end
    return nothing
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

    aux = AdaptiveComputation()
    return SeqPFChain(query, proc, state, aux)
end
