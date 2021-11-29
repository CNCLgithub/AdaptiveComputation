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


# returns the sensitivity of each latent variable
function hypothesize!(chain::SeqPFChain, att::MapSensitivity)

    @unpack proc, state, auxillary = chain
    @unpack objective, scale, samples, jitter = att

    # @show state.log_weights

    anyinf = @>> (state.traces) begin
            map(get_score)
            findfirst(isinf)
    end
    if !isnothing(anyinf)
        @>> (state.traces[anyinf]) begin
                get_retval
                last
                last
                display
        end
    end

    # sensitivities = @>> (state.traces) first n_obs zeros
    np = length(state.traces)
    arrousal = Vector{Float64}(undef, np)
    sensitivities = Dict{Int64, Vector{Float64}}()
    accepted = 0
    @inbounds for i = 1:np
        latents = trackers(state.traces[i])
        nl = length(latents)
        # base steps per tracker
        p_base_steps = nl == 0 ? 0 : floor(Int64, samples / nl) 
        seed_obj = objective(state.traces[i])
        # scs = correspondence(state.traces[i])
        psense = fill(-Inf, nl)
        for j = 1:nl, _ = 1:p_base_steps
            # println("Working on sample $(i), latent $(j)")
            jittered, ls = jitter(state.traces[i], latents[j] , att)
            jobj = objective(jittered)
            # jcs = correspondence(jittered)
            # jc = jcs[:, j]
            # compute sensitivity
            div = @>> jobj begin
                x -> sinkhorn_div(seed_obj, x; scale = scale)
                # log
            end
            psense[j] = logsumexp(psense[j], div)
            # marginal of assignment for latent to xs
            # c = jc .+ scs[:, j]
            # c .*= 0.5 * exp(div)
            # sensitivities += c

            # accept traces
            if log(rand()) < ls
                accepted +=1
                state.traces[i] = jittered
                seed_obj = jobj
                # scs = jcs
            end

        end
        sensitivities[i] = psense .- log(p_base_steps)
        arrousal[i] = nl == 0 ? -Inf : logsumexp(sensitivities[i]) 
    end
    println("acceptance ratio $(accepted / (np * samples))")
    # think about normalizing wrt to |xs|
    # sensitivities = log.(sensitivities) .- log(np * samples)

    @pack! auxillary = sensitivities
    @pack! auxillary = arrousal
    return nothing
end

# makes sensitivity weights smoother and softmaxes for categorical sampling
function goal_relevance!(chain::SeqPFChain, att::MapSensitivity)
    @unpack auxillary = chain
    @unpack sensitivities = auxillary
    weights = Dict{Int64, Vector{Float64}}()
    @inbounds for i = 1:length(sensitivities)
        nl = length(sensitivities[i])
        weights[i] = nl == 0 ? Float64[] : softmax(sensitivities[i] .* att.smoothness)
    end

    @pack! auxillary = weights
end

# returns number of sweeps (MH moves) to make determined
# by the sensitivity weights using an exponential function
function budget_cycles!(chain::SeqPFChain, att::MapSensitivity)
    @unpack auxillary = chain
    @unpack arrousal, cycles = auxillary
    @unpack sweeps, k, x0 = att
    m = sweeps / x0
    np = length(arrousal)
    cycles = Vector{Int64}(undef, np)
    @inbounds for i = 1:np
        amp = m * (arrousal[i] + x0)
        cycles[i] = @> amp begin
            clamp(0., sweeps)
            floor
            Int64
        end
    end
    println("avg cycles: $(mean(cycles))")
    # x = logsumexp(arrousal) - log(length(arrousal))
    # amp = k * (x + x0)

    # println("x: $(x), amp: $(amp)")
    # cycles = @> amp begin
    #     clamp(0., sweeps)
    #     floor
    #     Int64
    # end
    @pack! auxillary = cycles
    return nothing
end

# no early stopping for sensitivity-based attention
function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    false
end

