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

function trackers(tr::Gen.Trace)
    t, _, dm, _ = get_args(tr)
    trackers(dm, tr)
end

function register_to_obs!(gr::Vector{Float64}, tr::Gen.Trace, j::Int64, div::Float64)
    ws = correspondence(tr)
    ws = ws[:, j]
    ws .-= softmax(ws)
    rmul!(ws, div)
    gr += ws
    return nothing
end


# returns the sensitivity of each latent variable
function hypothesize!(chain::SeqPFChain, att::MapSensitivity)

    @unpack proc, state, auxillary = chain
    @unpack objective, scale, samples, jitter = att

    seeds = Gen.sample_unweighted_traces(state, samples)
    seed_ls = get_score.(seeds)
    seed_ls .-= maximum(seed_ls)
    # seed_obj = map(objective, seeds)
    # tracker_addrs = map(trackers, seeds)
    # n_latents = map(count_trackers, seeds)

    sensitivities = @>> seeds first n_obs zeros
    @inbounds for i = 1:samples
        latents = trackers(seeds[i])
        seed_obj = objective(seeds[i])
        for j = 1:length(latents)
            # println("Working on sample $(i), latent $(j)")
            jittered, ls = jitter(seeds[i], latents[j] , att)
            div = @>> jittered begin
                objective
                x -> sinkhorn_div(seed_obj, x; scale = scale)
                log
            end
            div += ls + seed_ls[i]
            register_to_obs!(sensitivities, jittered, j, div)
        end
    end

    println("log sensitivity")
    display(sensitivities)
    @pack! auxillary = sensitivities
    return nothing
end

# makes sensitivity weights smoother and softmaxes for categorical sampling
function goal_relevance!(chain::SeqPFChain, att::MapSensitivity)
    @unpack auxillary = chain
    @unpack sensitivities = auxillary
    weights = att.smoothness * sensitivities
    weights = softmax(weights)
    @pack! auxillary = weights
end

# returns number of sweeps (MH moves) to make determined
# by the sensitivity weights using an exponential function
function budget_cycles!(chain::SeqPFChain, att::MapSensitivity)
    @unpack auxillary = chain
    @unpack sensitivities = auxillary
    @unpack sweeps, k, x0 = att
    x = logsumexp(sensitivities) - log(length(sensitivities))
    amp = exp(-k * (x - x0))
    # amp = k * (x - x0)

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

