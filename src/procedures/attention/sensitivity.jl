import PhysicalConstants.CODATA2018: k_B
using Base.Iterators: take

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
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end

function get_stats(att::MapSensitivity, state::Gen.ParticleFilterState)
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
    gs = Vector{Float64}(undef, n_latents)
    display(kls)
    display(lls)
    for i = 1:n_latents
        weights = exp.((lls[:, i] .- logsumexp(lls[:, i])))
        # weights = min.(exp.(lls[:, i]), 1.0) # CHANGED no need to normalize
        gs[i] = sum(kls[:, i] .* weights)
    end
    gs = log.(gs)
    println("weights: $(gs)")
    return gs
end

function get_weights(att::MapSensitivity, stats)
    # # making it smoother
    gs = att.smoothness.*stats
    println("smoothed weights: $(gs)")
    softmax(gs)
end

function get_sweeps(att::MapSensitivity, stats)
    x = logsumexp(stats)
    # amp = att.sweeps / (1.0 + exp(-att.k*(x + att.x0)))
    # amp = att.k*(x - att.x0) + att.sweeps
    amp = att.k*x + att.x0
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
    record = AssociationRecord(100)
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
    kl = 0
    for i = 1:length(labels)
        kl += 0.5 * exp(probs[i, 1]) * (probs[i, 1] - ms[i])
        kl += 0.5 * exp(probs[i, 2]) * (probs[i, 2] - ms[i])
    end
    if kl < 0
        println("WATCH OUT!!!! kl $kl below 0")
    end
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
