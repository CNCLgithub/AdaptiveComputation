using Base.Iterators: take
using Distributions
using LinearAlgebra

export MapSensitivity

@with_kw struct MapSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    jitter::Function = tracker_kernel
    samples::Int = 1
    sweeps::Int = 5
    smoothness::Float64 = 1.003
    scale::Float64 = 1.0
    k::Float64 = 0.5
    x0::Float64 = 5.0
    ancestral_steps::Int = 3
    uniform_sweeps::Int = 0
    weights_tau::Float64 = 0.5 # proportion of old weights to keep
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end

function top_n_traces(state::Gen.ParticleFilterState, n::Int64)
    @unpack traces, log_weights = state
    inds = sortperm(log_weights; rev=true)[1:n]
    traces[inds]
end

# returns the sensitivity of each latent variable
function hypothesize!(chain::SeqPFChain, att::MapSensitivity)

    @unpack proc, state, auxillary = chain
    @unpack latents = proc
    @unpack sensitivities = auxillary
    @unpack objective, scale, samples, jitter = att

    seeds = Gen.sample_unweighted_traces(state, samples)
    # seeds = top_n_traces(state, samples)
    seed_obj = map(objective, seeds)

    n_latents = length(latents)

    kls = zeros(samples, n_latents)
    lls = zeros(samples, n_latents)
    @inbounds for i = 1:samples, j = 1:n_latents
        # println("Working on sample $(i), latent $(j)")
        jittered, lls[i, j] = jitter(seeds[i], j, att)
        kls[i, j] = @>> jittered begin
            objective
            sinkhorn_div(seed_obj[i]; scale = scale)
            log
        end
    end

    println("log kl")
    display(kls)
    println("log weights")
    display(lls)
    
    gs = Vector{Float64}(undef, n_latents)
    # normalizing accross samples
    clamp!(lls, -Inf, 0.)
    lses = @>> lls eachcol map(logsumexp)
    @inbounds for i = 1:n_latents
        # gs[i] = logsumexp(kls[:, i]) - log(samples)
        if isinf(lses[i]) || isnan(lses[i])
            gs[i] = logsumexp(kls[:, i]) - log(samples)
        else
            gs[i] = logsumexp(kls[:, i] + (lls[:, i] .- lses[i]))
        # gs[i] = logsumexp(kls[:, i] + lls[:, i]) - log(samples)
        end
    end
    println("compute weights: $gs")

    @unpack weights_tau = att
    inv_wt = 1.0 - weights_tau
    @inbounds for i in eachindex(sensitivities)
        # sensitivities[i] = isinf(sensitivities[i]) ? gs[i] : logsumexp(sensitivities[i] + weights_tau,
        #                                                              gs[i] + inv_wt)
        sensitivities[i] = isinf(sensitivities[i]) ? gs[i] : sensitivities[i] * weights_tau +
            gs[i] * inv_wt
    end
    @pack! auxillary = sensitivities
    println("time-smoothed weights: $(sensitivities)")
end

# makes sensitivity weights smoother and softmaxes for categorical sampling
function goal_relevance!(chain::SeqPFChain, att::MapSensitivity)
    @unpack auxillary = chain
    @unpack sensitivities = auxillary
    weights = att.smoothness * sensitivities
    weights = softmax(weights)
    println("sampling weights: $(weights)")
    @pack! auxillary = weights
end

# returns number of sweeps (MH moves) to make determined
# by the sensitivity weights using an exponential function
function budget_cycles!(chain::SeqPFChain, att::MapSensitivity)
    @unpack auxillary = chain
    @unpack sensitivities = auxillary
    @unpack sweeps, k, x0 = att
    x = logsumexp(sensitivities)
    # amp = sweeps / (1 + exp(-k*(x - x0)))
    amp = k * (x - x0)

    println("x: $(x), amp: $(amp)")
    cycles = @> amp begin
        clamp(0., sweeps)
        floor
        Int64
    end
    # println("cycles: $cycles")
    @pack! auxillary = cycles
    return nothing
end

# no early stopping for sensitivity-based attention
function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    false
end

