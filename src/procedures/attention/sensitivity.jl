import PhysicalConstants.CODATA2018: k_B
using Base.Iterators: take
using Distributions
using LinearAlgebra

export MapSensitivity

function jitter(tr::Gen.Trace, tracker::Int)
    args = Gen.get_args(tr)
    t = first(args)
    diffs = Tuple(fill(NoChange(), length(args)))
    addrs = []
    for i = max(1, t-3):t
        addr = :kernel => i => :dynamics => :brownian => tracker
        push!(addrs, addr)
    end
    (new_tr, ll) = take(regenerate(tr, args, diffs, Gen.select(addrs...)), 2)
end

function retrieve_latents(tr::Gen.Trace)
    args = Gen.get_args(tr)
    ntrackers = last(args).n_trackers
    collect(1:ntrackers)
end

@with_kw struct MapSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    latents::Function = t -> retrieve_latents(t)
    jitter::Function = jitter
    samples::Int = 1
    sweeps::Int = 5
    smoothness::Float64 = 1.003
    scale::Float64 = 100.0
    k::Float64 = 0.5
    x0::Float64 = 5.0
    ancestral_steps::Int = 3
    uniform_sweeps::Int = 0
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end


function pos_objective(traces, weights, tracker)

    positions = Matrix{Float64}(undef, 2, length(traces))
    for i=1:length(traces)
        (init_state, states) = Gen.get_retval(traces[i])
        trackers = states[end].graph.elements
        positions[1:2,i] = trackers[tracker].pos[1:2]
    end
    
    display(positions)
    display(weights)
    distribution = fit_mle(MvNormal, positions, weights)
    return mean(distribution), cov(distribution)
end

"""
    KL between multivariate Gaussian distributions
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
"""
function kl_mv_normal(mu_1, mu_2, sigma_1, sigma_2)
    # display(mu_1)
    # display(mu_2)
    # display(sigma_1)
    # display(sigma_2)
    if length(mu_1) != length(mu_2)
        error("dimensions must match!")
    end
    
    sigma_2_inv = inv(sigma_2)
    dim = length(mu_1)
    0.5*(
         tr(sigma_2_inv * sigma_1)
         + transpose(mu_2 - mu_1)*sigma_2_inv*(mu_2 - mu_1)
         - dim
         + log(det(sigma_2)/det(sigma_1))
        )
end

function get_categorical_weights(log_weights)
    (_, log_normalized_weights) = normalize_weights(log_weights)
    weights = exp.(log_normalized_weights)
end

function get_stats(att::MapSensitivity, state::Gen.ParticleFilterState)

    # attending to position
    if att.objective == pos_objective
        latents = att.latents(first(state.traces))
        
        # gets positions of trackers in terms of multivariate normal distributions
        weights = get_categorical_weights(state.log_weights)
        positions = map(x->pos_objective(state.traces, weights, x), latents)
        
        n_latents = length(latents)
        kls = zeros(att.samples, n_latents)
        lls = zeros(att.samples, n_latents)
        
        for i = 1:att.samples
            for j = 1:n_latents
                # sampling one unweighted trace
                weights = get_categorical_weights(state.log_weights)
                trace_id = Gen.categorical(weights)
                seed = state.traces[trace_id]
                
                # jittering that trace and computing new positons
                jittered, lls[i,j] = att.jitter(seed, j)

                new_traces = deepcopy(state.traces)
                new_traces[trace_id] = jittered
                
                log_weights = deepcopy(state.log_weights)
                log_weights[trace_id] = log_weights[trace_id] + lls[i,j] # TODO is this correct???????
                new_weights = get_categorical_weights(log_weights)

                new_position = pos_objective(new_traces, new_weights, j)

                kls[i,j] = kl_mv_normal(positions[j][1], new_position[1],
                                        positions[j][2], new_position[2])
            end
        end
        gs = Vector{Float64}(undef, n_latents)
        display(kls)
        display(lls)
        for i = 1:n_latents
            weights = exp.(min(zeros(att.samples), lls[:, i]))
            gs[i] = log(sum(kls[:, i] .* weights))
        end
        println("weights: $(gs)")
        return gs
        
    # attending to TD or DC
    else
        seeds = Gen.sample_unweighted_traces(state, att.samples)
        latents = att.latents(first(seeds))
        seed_obj = map(att.objective, seeds)
        n_latents = length(latents)
        kls = zeros(att.samples, n_latents)
        lls = zeros(att.samples, n_latents)
        for i = 1:att.samples
            jittered, ẟh = zip(map(idx -> att.jitter(seeds[i], idx),
                                   latents)...)
            lls[i, :] = collect(ẟh)
            jittered_obj = map(att.objective, jittered)
            ẟs = map(j -> relative_entropy(seed_obj[i], j),
                          jittered_obj)
            kls[i, :] = collect(ẟs)
        end
        display(kls)
        display(lls)
        gs = Vector{Float64}(undef, n_latents)
        lse = Vector{Float64}(undef, n_latents)
        for i = 1:n_latents
            lse[i] = logsumexp(lls[:, i])
            gs[i] = logsumexp(log.(kls[:, i]) .+ lls[:, i])
            # gs[i] =  logsumexp(log.(kls[:, i]) .+ lls[:, i]) - log(att.samples)
        end
        gs = gs .+ (lse .- logsumexp(lse))
        println("weights: $(gs)")
        return gs
    end
end

function get_weights(att::MapSensitivity, stats)
    # # making it smoother
    gs = att.smoothness.*stats
    println("smoothed weights: $(gs)")
    softmax(gs)
end

function get_sweeps(att::MapSensitivity, stats)
    x = logsumexp(stats)
    # amp = att.x0 * exp(-(x - att.k)^2 / (2*att.scale^2))
    # amp = att.x0 - att.k*(1 - exp(att.scale*x))
    amp = att.x0*exp(att.k*x)
    println("x: $(x), amp: $(amp)")
    Int64(round(clamp(amp, 0.0, att.sweeps)))
    # sweeps = min(att.sweeps, sum(stats))
    # round(Int, sweeps)
end

function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    # norm(new_stats) <= att.eps
    false
end

# Objectives


function _td(tr::Gen.Trace, t::Int)
    xs = get_choices(tr)[:kernel => t => :masks]
    pmbrfs = Gen.get_retval(tr)[2][t].rfs
    record = AssociationRecord(200)
    Gen.logpdf(rfs, xs, pmbrfs, record)
    tracker_assocs = map(c -> Set(vcat(c[2:end]...)), record.table)
    unique_tracker_assocs = unique(tracker_assocs)
    td = Dict{Set{Int64}, Float64}()
    for tracker_assoc in unique_tracker_assocs
        idxs = findall(map(x -> x == tracker_assoc, tracker_assocs))
        td[tracker_assoc] = logsumexp(record.logscores[idxs])
    end
    td
end

function target_designation(tr::Gen.Trace)
    k = first(Gen.get_args(tr))
    current_td = _td(tr, k)
end

function _dc(tr::Gen.Trace, t::Int64,  scale::Float64)
    xs = get_choices(tr)[:kernel => t => :masks]
    pmbrfs = Gen.get_retval(tr)[2][t].rfs
    record = AssociationRecord(100)
    Gen.logpdf(rfs, xs, pmbrfs, record)
    Dict{Vector{Vector{Int64}}, Float64}(zip(record.table,
                                             record.logscores ./ scale))
end

function data_correspondence(tr::Gen.Trace; scale::Float64 = 1.0)
    k = first(Gen.get_args(tr))
    d = _dc(tr, k, scale)
end


# Helpers

"""
Computes the entropy of a discrete distribution
"""
function entropy(ps::AbstractArray{Float64})
    # -k_B * sum(map(p -> p * log(p), ps))
    normed = ps .- logsumexp(ps)
    s = 0
    for (p,n) in zip(ps, normed)
        s += p * exp(n)
    end
    -1 * s
end

function entropy(pd::Dict)
    lls = collect(Float64, values(pd))
    log(entropy(lls))
end

function resolve_correspondence(p::T, q::T) where T<:Dict
    s = collect(intersect(keys(p), keys(q)))
    vals = Matrix{Float64}(undef, length(s), 2)
    for (i,k) in enumerate(s)
        vals[i, 1] = p[k]
        vals[i, 2] = q[k]
    end
    (s, vals)
end


function relative_entropy(p::T, q::T) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error("empty intersect")
    end
    probs[:, 1] .-= logsumexp(probs[:, 1])
    probs[:, 2] .-= logsumexp(probs[:, 2])
    ms = collect(map(logsumexp, eachrow(probs))) .- log(2)
    # display(p); display(q)
    order = sortperm(probs[:, 1], rev= true)
    # display(Dict(zip(labels[order], eachrow(probs[order, :]))))
    # println("new set")
    kl = 0.0
    for i in order
        _kl = 0.0
        _kl += 0.5 * exp(probs[i, 1]) * (probs[i, 1] - ms[i])
        _kl += 0.5 * exp(probs[i, 2]) * (probs[i, 2] - ms[i])
        kl += isnan(_kl) ? 0.0 : _kl
        # println("$(labels[i]) => $(probs[i, :]) | kl = $(kl)")

    end
    isnan(kl) ? 0.0 : clamp(kl, 0.0, 1.0)
end

function index_pairs(n::Int)
    if !iseven(n)
        n -= 1
    end
    indices = shuffle(collect(1:n))
    reshape(indices, Int(n/2), 2)
end

function get_dims(latents::Function, trace::Gen.Trace)
    results = latents(trace)
    size(results, 2)
end
