export AbstractAttentionModel,
    hypothesis_testing!,
    update_importance!,
    update_arrousal!,
    adaptive_compute!

abstract type AbstractAttentionModel end

function hypothesis_testing!(::InferenceChain, ::AbstractAttentionModel)
    error("not implemented")
end

function update_importance!(::InferenceChain, ::AbstractAttentionModel)
    error("not implemented")
end

function update_arrousal!(::InferenceChain, ::AbstractAttentionModel)
    error("not implemented")
end

function adaptive_compute!(chain::PFChain,
                           attention::AbstractAttentionModel)
    # apply perceptual updates and monitor dP/dS
    @time hypothesis_testing!(chain, attention)
    update_importance!(chain, attention)
    update_arrousal!(chain, attention)
    return nothing
end

include("objectives.jl")
include("distances.jl")
# include("uniform.jl")
include("sensitivity.jl")
include("perturb_state.jl")
