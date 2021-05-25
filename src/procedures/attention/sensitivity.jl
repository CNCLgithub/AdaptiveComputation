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
    ntrackers = args[2].n_trackers
    collect(1:ntrackers)
end

@with_kw mutable struct MapSensitivity <: AbstractAttentionModel
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
    weights::Union{Vector{Float64}, Nothing} = nothing
    weights_tau::Float64 = 0.5
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end

function get_stats(att::MapSensitivity, state::Gen.ParticleFilterState)

    seeds = Gen.sample_unweighted_traces(state, att.samples)
    seed_obj = map(att.objective, seeds)

    latents = att.latents(first(seeds))
    n_latents = length(latents)

    kls = zeros(att.samples, n_latents)
    lls = zeros(att.samples, n_latents)
    for i = 1:att.samples
        jittered, ẟh = @>> latents begin
            map(idx -> att.jitter(seeds[i], idx))
            x -> zip(x...)
        end
        lls[i, :] = collect(ẟh)
        jittered_obj = map(att.objective, jittered)
        # ẟs = @>> jittered_obj map(j -> relative_entropy(seed_obj[i], j))
        ẟs = @>> jittered_obj map(j -> l2_d(seed_obj[i], j))
        kls[i, :] = collect(ẟs)
    end
    
    println("log.(kls)")
    display(log.(kls))
    println("kls weights (exp)")
    display(exp.(lls))
    
    gs = Vector{Float64}(undef, n_latents)
    lse = Vector{Float64}(undef, n_latents)
    # normalizing accross samples
    clamp!(lls, -Inf, 0.)
    for i = 1:n_latents
        lse[i] = logsumexp(lls[:, i])
        # gs[i] = logsumexp(log.(kls[:, i]) .+ lls[:, i]) - log(att.samples)
        gs[i] = logsumexp(log.(kls[:, i]) .+ lls[:, i] .- lse[i])
    end
    println("compute weights: $gs")
    # normalizing accross trackers for unstable version
    # gs = gs .+ (lse .- logsumexp(lse))
    if isnothing(att.weights)
        att.weights = gs
    else
        att.weights = log.(
            exp.(att.weights .+ log(att.weights_tau)) .+
            exp.(gs .+ log(1.0 - att.weights_tau)))
    end
    println("time-smoothed weights: $(att.weights)")
    return att.weights
end

function get_weights(att::MapSensitivity, stats)
    # mean_weights = fill(mean(att.weights), length(att.weights))
    # weights = att.smoothness*mean_weights + (1.0 - att.smoothness)*att.weights
    weights = att.smoothness * stats
    weights = softmax(weights)
    println("sampling weights: $(weights)")
    weights
end

function get_sweeps(att::MapSensitivity, stats)
    x = logsumexp(stats)
    #amp = att.sweeps*exp(att.k*min(x, 0.0))
    amp = att.sweeps*exp(att.k*(x - att.x0))

    println("k: $(att.k)")
    println("x: $(x), amp: $(amp)")
    sweeps = Int64(round(clamp(amp, 0.0, att.sweeps)))
    println("sweeps: $sweeps")
    return sweeps
end

function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    # norm(new_stats) <= att.eps
    false
end

# Objectives

function _td2(xs::Vector{BitArray}, pmbrfs::RFSElements, t::Int)
    record = AssociationRecord(200)
    Gen.logpdf(rfs, xs, pmbrfs, record)
    @assert first(pmbrfs) isa PoissonElement

    tracker_assocs = @>> (record.table) begin
        map(c -> Set(vcat(c[2:end]...)))
    end
    k = length(xs)
    denom = logsumexp(record.logscores)
    td = Dict{Int64, Float64}()
    for x = 1:k
        idxs = findall(map(es -> in(x, es), tracker_assocs))
        td[x] = logsumexp(record.logscores[idxs])
    end
    td
end

function _td(xs::Vector{BitArray}, pmbrfs::RFSElements, t::Int)
    record = AssociationRecord(200)
    Gen.logpdf(rfs, xs, pmbrfs, record)
    @assert first(pmbrfs) isa PoissonElement

    tracker_assocs = @>> (record.table) begin
        map(c -> Set(vcat(c[2:end]...)))
    end
    unique_tracker_assocs = unique(tracker_assocs)
    td = Dict{Set{Int64}, Float64}()
    for tracker_assoc in unique_tracker_assocs
        idxs = findall(map(x -> x == tracker_assoc, tracker_assocs))
        td[tracker_assoc] = logsumexp(record.logscores[idxs])
    end
    td
end

function target_designation_receptive_fields(tr::Gen.Trace)
    t = first(Gen.get_args(tr))

    rfs_vec = @>> Gen.get_retval(tr) begin
        last # get the states
        last # get the last state
        (cg -> get_prop(cg, :rfs_vec)) # get the receptive fields
    end # rfes for each rf

    receptive_fields = @> tr begin
        get_choices
        get_submap(:kernel => t => :receptive_fields)
        get_submaps_shallow
        # vec of tuples (rf id, rf mask choicemap)
    end # masks for each rf

    # @debug "receptive fields $(typeof(receptive_fields[1]))"
    tds = @>> receptive_fields begin
        map(rf -> _td2(convert(Vector{BitArray}, rf[2][:masks]), rfs_vec[rf[1]], t))
    end
end

function target_designation(tr::Gen.Trace)
    t = first(Gen.get_args(tr))
    xs = get_choices(tr)[:kernel => t => :masks]
    pmbrfs = Gen.get_retval(tr)[2][t].rfs

    current_td = _td(xs, pmbrfs, t)
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

# this is for receptive fields
function relative_entropy(ps::T, qs::T) where T<:Array
    @>> zip(ps, qs) begin
        map(x -> relative_entropy(x[1], x[2]; error_on_empty=false))
        mean
    end
end

function relative_entropy(p::T, q::T;
                          error_on_empty=true) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error_on_empty && error("empty intersect")
        return 1.0
    end
    println("probs = ")
    display(Dict(zip(labels, eachrow(probs))))
    display(logsumexp(probs[:, 1]))
    display(logsumexp(probs[:, 2]))
    display(typeof(probs))
    probs[:, 1] .-= logsumexp(probs[:, 1])
    probs[:, 2] .-= logsumexp(probs[:, 2])
    ms = collect(map(logsumexp, eachrow(probs))) .- log(2)
    #display(p); display(q)
    order = sortperm(probs[:, 1], rev= true)
    #display(Dict(zip(labels[order], eachrow(probs[order, :]))))
    kl = 0.0
    for i in order
        _kl = 0.5 * exp(probs[i, 1]) * (probs[i, 1] - ms[i])
        _kl += 0.5 * exp(probs[i, 2]) * (probs[i, 2] - ms[i])
        kl += isnan(_kl) ? 0.0 : _kl
        println("$(labels[i]) => $(probs[i, :]) | $(ms[i]) => kl = $(log(kl))")
    end
    isnan(kl) ? 0.0 : clamp(kl, 0.0, 1.0)
end

function jeffs_d(ps::T, qs::T) where T<:Array
    @>> zip(ps, qs) begin
        map(x -> jeffs_d(x[1], x[2]; error_on_empty=false))
        mean
    end
end

function jeffs_d(p::T, q::T;
                 error_on_empty=true) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error_on_empty && error("empty intersect")
        return 1.0
    end
    println("probs = ")
    display(Dict(zip(labels, eachrow(probs))))
    display(logsumexp(probs[:, 1]))
    display(logsumexp(probs[:, 2]))
    probs[:, 1] .-= logsumexp(probs[:, 1])
    probs[:, 2] .-= logsumexp(probs[:, 2])
    order = sortperm(probs[:, 1], rev= true)
    jd = 0.0
    for i in order
        _jd = exp(probs[i, 1]) - exp(probs[i, 2])
        _jd *= (probs[i, 1] - probs[i, 2])
        jd += _jd
        println("$(labels[i]) => $(probs[i, :]) => jd = $(log(jd))")
    end
    return jd
    # isnan(kl) ? 0.0 : clamp(kl, 0.0, 1.0)
end

function l2_d(ps::T, qs::T) where T<:Array
    @>> zip(ps, qs) begin
        map(x -> l2_d(x[1], x[2]; error_on_empty=false))
        mean
    end
end


function l2_d(p::T, q::T;
                 error_on_empty=true) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error_on_empty && error("empty intersect")
        return 1.0
    end
    println("probs = ")
    display(Dict(zip(labels, eachrow(probs))))
    norm(probs[:, 1] - probs[:, 2])
    # norm(exp.(probs[:, 1]) - exp.(probs[:, 2]))
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
