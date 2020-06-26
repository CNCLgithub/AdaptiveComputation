export UniformAttention

@with_kw struct UniformAttention <: AbstractAttentionModel
    sweeps::Int = 0
    perturb_function::Union{Function, Nothing} = nothing
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
