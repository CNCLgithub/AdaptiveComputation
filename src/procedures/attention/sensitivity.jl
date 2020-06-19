using PhysicalConstants

export PairwiseSensitivity

@with_kw struct PairwiseSensitivity <: AbstractAttentionModel
    objective::Function = target_designation
    latents::Function = extract_tracker_positions
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
    n_pairs = length(indices)
    n_latents = get_dims(att.objective, state.traces[1])
    gradients = zeros(n_pairs, n_latents)
    weights = zeros(n_pairs)
    for i = 1:n_pairs
        weights[i] = exp(sum(state.log_weights[indices[i]]))
        traces = state.traces[indices[i]]
        entropies = map(entropy ∘ objective, traces)
        latents = map(att.latents, traces)
        ẟs = diff(entropies)
        ẟh = norm.(diff(latents, dims = 1))
        gradients[i, :] = ẟs./ẟh
    end
    abs.(gradients) .* weights ./ sum(weights)
end

function get_sweeps(att::PairwiseSensitivity, stats)
    return att.sweeps
end

function early_stopping(att::PairwiseSensitivity, new_stats, prev_stats)
    # norm(new_stats) <= att.eps
    false
end

# Objectives

function target_designation(tr::Gen.trace)
    pmbrfs_stats = Gen.get_retval(tr)[2][end].pmbrfs_params.pmbrfs_stats
    exp.(pmbrfs_stats.ll)
end


# Helpers

"""
Computes the entropy of a discrete distribution
"""
function entropy(ps::AbstractArray{Float64})
    -k_B * sum(map(p -> p * log(p), ps))
end

function index_pairs(n::Int)
    if !iseven(n)
        n -= 1
    end
    indices = shuffle(collect(1:n))
    reshape(indices, n/2, 2)
end

function get_dims(objective::Function, trace::Gen.Trace)
    results = objective(trace)
    size(results, 1)
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
