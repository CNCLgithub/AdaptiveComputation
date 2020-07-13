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
    seed_obj = map(att.objective, seeds)
    n_latents = size(first(latents), 1)
    kls = zeros(att.samples, n_latents)
    lls = zeros(att.samples, n_latents)
    for i = 1:att.samples
        jittered, ẟh = zip(map(idx -> att.jitter(seeds[i], idx),
                                    1:n_latents)...)
        lls[i, :] = collect(ẟh)
        jittered_obj = map(att.objective, jittered)
        ẟs = abs.(map(j -> relative_entropy(seed_obj[i], j),
                      jittered_obj))
        kls[i, :] = collect(ẟs)
    end
    gs = Vector{Float64}(undef, n_latents)
    display(kls)
    display(lls)
    for i = 1:n_latents
        gs[i] = sum(kls[:, i] .* exp.(lls[:, i] .- logsumexp(lls[:, i])))
        # gs[i] = sum(kls[:, i]) /att.samples
        # gs[i] = sum(kls[:, i] .* exp.(lls[:, i]))
    end
    println("kl per tracker: $(gs)")
    gs
    # log.(gs)
end

function get_sweeps(att::MapSensitivity, stats)
    # sweeps = min(att.sweeps, sum(stats))
    # round(Int, sweeps)
    # println(logsumexp(stats))
    # x = logsumexp(stats)
    # # amp = att.sweeps * (att.m)^logsumexp(stats)
    # # amp = 0.5(x - -30.0) + 1
    # amp = (15.) / (1 + exp(-1*0.15(x - (-23.0))))
    x = sum(stats)
    amp = x < 300 ? 0 : 15.0*x / 1600.0
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
                            scale::Float64 = 1.0)
    k = first(Gen.get_args(tr))
    current_td = _td(tr, k, scale)
    previous = []
    for t = max(1, k-w):(k - 1)
        push!(previous, _td(tr, t, scale))
    end
    Base.merge((x,y) -> 0.5*(x+y), current_td, previous...)
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
    -1 * sum(map(p -> p * exp(p), ps))
end

function entropy(pd::Dict)
    lls = collect(Float64, values(pd))
    entropy(lls)
end

function extend(as, bs, reserve)
    not_in_p = setdiff(bs, as)
    es = []
    for k in not_in_p
        push!(es, (k, reserve))
    end
    return Dict(es)
end


function _merge(p::T, q::T) where T<:Dict
    keys_p, lls_p = zip(p...)
    reserve_p =  last(lls_p) * 1.1 #log(1 - sum(exp.(lls_p))) - log(1E5)
    # reserve_p =  log(1 - sum(exp.(lls_p)))
    keys_q, lls_q = zip(q...)
    # reserve_q = log(1 - sum(exp.(lls_q)))
    reserve_q = last(lls_q) * 1.1 # logsumexp(lls_p)

    extended_p = Base.merge(p, extend(keys_p, keys_q, reserve_p))
    extended_q = Base.merge(q, extend(keys_q, keys_p, reserve_q))
    (extended_p, extended_q)
end

function relative_entropy(p::T, q::T) where T<:Dict
    ps, qs = _merge(p, q)
    p_den = logsumexp(collect(Float64, values(ps)))
    q_den = logsumexp(collect(Float64, values(qs)))
    kl = 0
    # println(ps)
    # println(qs)
    for (k, v) in ps
        kl += exp(v - p_den) * (v - qs[k])
    end
    # println(kl)
    # @assert kl >=  0
    kl
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
