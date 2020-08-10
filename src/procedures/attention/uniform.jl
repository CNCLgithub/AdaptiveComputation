export UniformAttention

@with_kw struct UniformAttention <: AbstractAttentionModel
    sweeps::Int = 0
end

function get_stats(::UniformAttention, state::Gen.ParticleFilterState)
    t, motion, gm = Gen.get_args(first(state.traces))
    return zeros(gm.n_trackers)
end

function get_sweeps(attention::UniformAttention, stats)
    return attention.sweeps
end

function early_stopping(::UniformAttention, new_stats, prev_stats)
    return false
end

function load(::Type{UniformAttention}, path::String, trial::Int, k::Int)
    trial_dir = joinpath(path, "$(trial)")
    trial_results = load_trial(trial_dir)
    sweeps = round(Int, mean(trial_results["compute"])/k)
    UniformAttention(;sweeps = sweeps)
end
