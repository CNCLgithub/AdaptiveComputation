import PhysicalConstants.CODATA2018: k_B
using Base.Iterators: take
using Distributions
using LinearAlgebra

export MapSensitivity

@with_kw mutable struct MapSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    latents::Function = t -> retrieve_latents(t)
    jitter::Function = tracker_kernel
    samples::Int = 1
    sweeps::Int = 5
    smoothness::Float64 = 1.003
    scale::Float64 = 100.0
    k::Float64 = 0.5
    x0::Float64 = 5.0
    ancestral_steps::Int = 3
    uniform_sweeps::Int = 0
    weights::Union{Vector{Float64}, Nothing} = nothing # used for averaging
    weights_tau::Float64 = 0.5 # proportion of old weights to keep
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end

function retrieve_latents(tr::Gen.Trace)
    args = Gen.get_args(tr)
    ntrackers = args[2].n_trackers
    collect(1:ntrackers)
end

# returns the sensitivity of each latent variable
function hypothesize!(state::Gen.ParticleFilterState, att::MapSensitivity)

    @unpack objective, samples, latents, jitter = att
    seeds = Gen.sample_unweighted_traces(state, samples)
    seed_obj = map(objective, seeds)

    latents = att.latents(first(seeds))
    n_latents = length(latents)

    kls = zeros(samples, n_latents)
    lls = zeros(samples, n_latents)
    @inbounds for i = 1:samples, j = 1:n_latents
        # println("Working on sample $(i), latent $(j)")
        jittered, lls[i, j] = jitter(seeds[i], j, att)
        kls[i, j] = @>> jittered begin
            objective
            js_div(seed_obj[i])
            # jeffs_d(seed_obj[i])
            log
        end
    end

    println("log kl")
    display(kls)
    println("log weights")
    display(lls)
    
    gs = Vector{Float64}(undef, n_latents)
    # lse = Vector{Float64}(undef, n_latents)
    # normalizing accross samples
    clamp!(lls, -Inf, 0.)
    for i = 1:n_latents
        # gs[i] = logsumexp(kls[:, i]) - log(samples)
        gs[i] = logsumexp(kls[:, i] + lls[:, i]) - log(samples)
        # gs[i] = sum(exp.(log.(kls[:, i]) .+ lls[:, i])) / att.samples
    end
    println("compute weights: $gs")

    if isnothing(att.weights) || any(isinf.(att.weights))
        att.weights = gs
    else
        # applying a smoothing kernel across time
        att.weights = (att.weights * att.weights_tau +
                       gs * (1.0 - att.weights_tau))
    end
    println("time-smoothed weights: $(att.weights)")
    return att.weights
end

# makes sensitivity weights smoother and softmaxes for categorical sampling
function get_weights(att::MapSensitivity, stats::Vector{Float64})
    weights = att.smoothness * stats
    weights = softmax(weights)
    println("sampling weights: $(weights)")
    weights
end

# returns number of sweeps (MH moves) to make determined
# by the sensitivity weights using an exponential function
function get_sweeps(att::MapSensitivity, stats)
    x = logsumexp(stats)
    # amp = att.sweeps*exp(att.k*(x - att.x0))
    amp = att.sweeps / (1 + exp(-att.k*(x - att.x0)))

    println("x: $(x), amp: $(amp)")
    sweeps = Int64(round(clamp(amp, 0.0, att.sweeps)))
    println("sweeps: $sweeps")
    return sweeps
end

# no early stopping for sensitivity-based attention
function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    false
end

