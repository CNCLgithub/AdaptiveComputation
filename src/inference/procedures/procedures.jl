using Gen_Compose
using Gen_Compose: initial_args, initial_constraints,
    AuxillaryState, SeqPFChain

@with_kw mutable struct AdaptiveComputation <: AuxillaryState
    acceptance::Float64 = 0.
    arrousal::Float64 = 0
    importance::Vector{Float64}
    sensitivities::Vector{Float64}
end

include("latent_maps.jl")
include("attention/attention.jl")
include("particle_filter.jl")
