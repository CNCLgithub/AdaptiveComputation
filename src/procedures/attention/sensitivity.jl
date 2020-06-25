import PhysicalConstants.CODATA2018: k_B
using Base.Iterators: take

export PairwiseSensitivity, MapSensitivity

function jitter(tr::Gen.Trace, tracker::Int)
    args = Gen.get_args(tr)
    t = first(args)
    diffs = Tuple(fill(NoChange(), length(args)))
    addr = :states => t => :dynamics => :brownian => tracker
    take(regenerate(tr, args, diffs, Gen.select(addr)), 2)
end

@with_kw struct MapSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    latents::Function = t -> extract_tracker_positions(t)[1, 1, :, :]
    jitter::Function = jitter
    samples::Int = 1
    sweeps::Int = 5
    eps::Float64 = 0.01
end

function load(::Type{MapSensitivity}, path; kwargs...)
    MapSensitivity(;read_json(path)..., kwargs...)
end

# function get_stats(att::MapSensitivity, state::Gen.ParticleFilterState)
#     seeds = Gen.sample_unweighted_traces(state, att.samples)
#     latents = map(att.latents, seeds)
#     seed_obj = map(entropy ∘ att.objective, seeds)
#     n_latents = size(first(latents), 2)
#     gradients = zeros(att.samples, n_latents)
#     for i = 1:att.samples
#         seed_latents = att.latents(seeds[i])
#         jittered, weights = zip(map(idx -> att.jitter(seeds[i], idx),
#                                     1:n_latents)...)
#         new_latents = map(att.latents, jittered)
#         # ∘ == 2218: ring operator
#         jittered_obj = map(entropy ∘ att.objective, jittered)
#         ẟs = seed_obj[i] .- jittered_obj
#         ẟs = abs.(ẟs)
#         ẟh = map(norm, eachrow(latents[i,:] .- new_latents))
#         # println(ẟs)
#         # println(ẟh)
#         gradients[i, :] = exp.(log.(ẟs) .- log.(ẟh))
#     end
#     gs = vec(abs.(mean(gradients, dims = 1)))
# end
function get_stats(att::MapSensitivity, state::Gen.ParticleFilterState)
    seeds = Gen.sample_unweighted_traces(state, att.samples)
    latents = map(att.latents, seeds)
    seed_obj = map(att.objective, seeds)
    n_latents = size(first(latents), 2)
    gradients = zeros(att.samples, n_latents)
    for i = 1:att.samples
        seed_latents = att.latents(seeds[i])
        jittered, weights = zip(map(idx -> att.jitter(seeds[i], idx),
                                    1:n_latents)...)
        new_latents = map(att.latents, jittered)
        jittered_obj = map(att.objective, jittered)
        ẟs = map(j -> relative_entropy(seed_obj[i], j), jittered_obj)
        ẟs = abs.(ẟs)
        ẟh = map(norm, eachrow(latents[i,:] .- new_latents))
        # println(ẟs)
        # println(ẟh)
        gradients[i, :] = exp.(log.(ẟs) .- log.(ẟh))
    end
    gs = vec(abs.(mean(gradients, dims = 1)))
end

function get_sweeps(att::MapSensitivity, stats)
    norm(stats) >= att.eps ? att.sweeps : 0
end

function early_stopping(att::MapSensitivity, new_stats, prev_stats)
    # norm(new_stats) <= att.eps
    false
end

@with_kw struct PairwiseSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    latents::Function = t -> extract_tracker_positions(t)[1, 1, :, :]
    sweeps::Int = 5
    eps::Float64 = 0.01
end

function load(::Type{PairwiseSensitivity}, path; kwargs...)
    PairwiseSensitivity(;read_json(path)..., kwargs...)
end


function get_stats(att::PairwiseSensitivity, state::Gen.ParticleFilterState)
    num_particles = length(state.traces)
    # using weighted traces
    indices = index_pairs(num_particles)
    n_pairs = size(indices, 1)
    n_latents = get_dims(att.latents, state.traces[1])
    gradients = zeros(n_pairs, n_latents)
    weights = zeros(n_pairs)
    for i = 1:n_pairs
        idxs = indices[i, :]
        weights[i] = sum(state.log_weights[idxs])
        traces = state.traces[idxs]
        entropies = map(entropy ∘ att.objective, traces)
        a_l, b_l = map(att.latents, traces)
        ẟs = diff(entropies)
        ẟh = max.(map(norm, eachrow(a_l - b_l)), 1E-10)
        gradients[i, :] = ẟs./ẟh
    end
    gs = vec(sum(abs.(gradients) .* weights ./ logsumexp(weights), dims = 1))
end

function get_sweeps(att::PairwiseSensitivity, stats)
    norm(stats) >= att.eps ? att.sweeps : 0
end

function early_stopping(att::PairwiseSensitivity, new_stats, prev_stats)
    # norm(new_stats) <= att.eps
    false
end

# Objectives


function _td(tr::Gen.Trace, t)
    ret = Gen.get_retval(tr)[2][t]
    tds = ret.pmbrfs_params.pmbrfs_stats.partitions
    lls = min.(ret.pmbrfs_params.pmbrfs_stats.ll, 1E-10)
    Dict(zip(tds, lls))
end

function target_designation(tr::Gen.Trace; w::Int = 3)
    k = first(Gen.get_args(tr))
    current_td = _td(tr, k)
    previous = []
    for t = max(1, k-w):(k - 1)
        push!(previous, _td(tr, t))
    end
    Base.merge((x,y) -> 0.5*(x+y), current_td, previous...)
end


# Helpers

"""
Computes the entropy of a discrete distribution
"""
function entropy(ps::AbstractArray{Float64})
    # -k_B * sum(map(p -> p * log(p), ps))
    println(ps)
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
    reserve_p = last(lls_p) + log(0.9) # logsumexp(lls_p)
    keys_q, lls_q = zip(q...)
    reserve_q = last(lls_q) + log(0.9) # logsumexp(lls_p)

    extended_p = Base.merge(p, extend(keys_p, keys_q, reserve_p))
    extended_q = Base.merge(q, extend(keys_q, keys_p, reserve_q))
    (extended_p, extended_q)
end

function relative_entropy(p::T, q::T) where T<:Dict
    ps, qs = _merge(p, q)
    p_den = logsumexp(collect(Float64, values(ps)))
    q_den = logsumexp(collect(Float64, values(qs)))
    kl = 0
    for (k, v) in ps
        # println("exp $v * $(qs[k]) - $(v)")
        kl += exp(v - p_den) * (qs[k] - v)
    end
    -1 * kl
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


# """
# Returns a weighted vector of approximated
# elbo derivatives per object
# """
# function elbo(s, ss, logs)
#     weights = logsumexp(logs)
#     ss .* weights - s
# end

# function first_order(state::Gen.ParticleFilterState, objective, args, cm)
#     map_tr = get_map(state)
#     current_s = (entropy ∘ objective)(map_tr)
#     prediction, p_ls = Gen.update(map_tr, args, (UnknownChange,), cm)
#     base_h = (entropy ∘ target_designation)(prediction)
#     perturbations = map(i -> perturb_state(prediction, i), 1:N)
#     trs, lgs = zip(perturbations...)
#     entropies = map(entropy ∘ target_designation, trs)
#     elbo(base_h, entropies, lgs)
# end
