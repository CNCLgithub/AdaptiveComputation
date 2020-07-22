import PhysicalConstants.CODATA2018: k_B
using Base.Iterators: take

export MapSensitivity

function jitter(tr::Gen.Trace, tracker::Int)
    args = Gen.get_args(tr)
    t = first(args)
    diffs = Tuple(fill(NoChange(), length(args)))
    addr = :states => t => :dynamics => :brownian => tracker
    (new_tr, ll) = take(regenerate(tr, args, diffs, Gen.select(addr)), 2)
    # (new_tr, get_score(new_tr))
    # (new_tr, min(ll, 0))
end

@with_kw struct MapSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    latents::Function = t -> extract_tracker_positions(t)[1, 1, :, :]
    jitter::Function = jitter
    samples::Int = 1
    sweeps::Int = 5
    scale::Float64 = 100.0
    m::Float64 = 1.04
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end

function get_stats(att::MapSensitivity, state::Gen.ParticleFilterState)
    seeds = Gen.sample_unweighted_traces(state, att.samples)
    latents = map(att.latents, seeds)
    # seed_obj = map(entropy ∘ att.objective, seeds)
    seed_obj = map(att.objective, seeds)
    n_latents = size(first(latents), 1)
    kls = zeros(att.samples, n_latents)
    lls = zeros(att.samples, n_latents)
    for i = 1:att.samples
        jittered, ẟh = zip(map(idx -> att.jitter(seeds[i], idx),
                                    1:n_latents)...)
        lls[i, :] = collect(ẟh)
        # ẟs = seed_obj[i] .- map(entropy ∘ att.objective, jittered)
        # kls[i, :] = collect(abs.(ẟs))
        jittered_obj = map(att.objective, jittered)
        ẟs = map(j -> relative_entropy(seed_obj[i], j),
                      jittered_obj)
        kls[i, :] = collect(ẟs)
    end
    gs = Vector{Float64}(undef, n_latents)
    display(kls)
    display(lls)
    for i = 1:n_latents
        # gs[i] = mean(kls[:, i])
        weights = exp.((lls[:, i] .- logsumexp(lls[:, i])))
        gs[i] = sum(kls[:, i] .* weights)
    end
    println("kl per tracker: $(gs)")
    log.(gs)
end

function get_sweeps(att::MapSensitivity, stats)
    x = logsumexp(stats)
    # g = x / 100.
    # amp = 15. * exp((x + 100)/245)
    amp = 15.0 / (1.0 + exp(-0.45(x + 13.2)))
    # amp = x < -1000 ? 0 : 5
    println("x: $(x), amp: $(amp)")
    round(Int, min(amp, att.sweeps))
end

function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    # norm(new_stats) <= att.eps
    false
end

# Objectives


function _td(tr::Gen.Trace, t::Int, scale::Float64)
    ret = Gen.get_retval(tr)[2][t]
    tds = ret.pmbrfs_params.pmbrfs_stats.partitions
    tds = map(first, tds)
    lls = ret.pmbrfs_params.pmbrfs_stats.ll_partitions ./ scale
    Dict(zip(tds, lls))
end

function target_designation(tr::Gen.Trace; w::Int = 0,
                            scale::Float64 = 100.0)
    k = first(Gen.get_args(tr))
    current_td = _td(tr, k, scale)
    previous = []
    for t = max(1, k-w):(k - 1)
        push!(previous, _td(tr, t, scale))
    end
    Base.merge((x,y) -> logsumexp([x,y]) .- log(2), current_td, previous...)
end

function _dc(tr::Gen.Trace, t::Int, scale::Float64)
    ret = Gen.get_retval(tr)[2][t]
    tds = ret.pmbrfs_params.pmbrfs_stats.assignments
    lls = ret.pmbrfs_params.pmbrfs_stats.ll_assignments ./ scale
    Dict(zip(tds, lls))
end

function data_correspondence(tr::Gen.Trace; scale::Float64 = 10.0)
    k = first(Gen.get_args(tr))
    d = _td(tr, k, scale)
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

function extend(as, bs, reserve)
    not_in_p = setdiff(bs, as)
    nn = length(not_in_p)
    per_missing = reserve - log(nn)
    es = []
    for k in not_in_p
        push!(es, (k, per_missing))
    end
    return Dict(es)
end


function _merge(p::T, q::T) where T<:Dict
    keys_p, lls_p = zip(p...)
    reserve_p =  minimum(lls_p) * 1.01
    keys_q, lls_q = zip(q...)
    reserve_q = minimum(lls_q) * 1.01

    extended_p = Base.merge(p, extend(keys_p, keys_q, reserve_p))
    extended_q = Base.merge(q, extend(keys_q, keys_p, reserve_q))
    vals = Matrix{Float64}(undef, length(extended_p), 2)
    # println(extended_p)
    # println(extended_q)
    for (i,k) in enumerate(keys(extended_p))
        vals[i, 1] = extended_p[k]
        vals[i, 2] = extended_q[k]
    end
    (keys(extended_p), vals)
end

function relative_entropy(p::T, q::T) where T<:Dict
    labels, probs = _merge(p, q)
    probs[:, 1] .-= logsumexp(probs[:, 1])
    probs[:, 2] .-= logsumexp(probs[:, 2])
    ms = log.(sum(exp.(probs), dims = 2)) .- log(2)
    # display(probs)
    # display(ms)
    kl = 0
    for i = 1:length(labels)
        kl += 0.5 * exp(probs[i, 1]) * (probs[i, 1] - ms[i])
        kl += 0.5 * exp(probs[i, 2]) * (probs[i, 2] - ms[i])
    end
    # println(kl)
    max(kl, 0)
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
