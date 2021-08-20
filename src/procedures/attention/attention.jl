export AbstractAttentionModel,
    get_stats,
    get_sweeps,
    early_stopping,
    rejuvenate_attention!

abstract type AbstractAttentionModel end

function hypothesize!(::InferenceChain, ::AbstractAttentionModel)
    error("not implemented")
end

function goal_relevance!(::InferenceChain, ::AbstractAttentionModel)
    error("not implemented")
end

function budget_cycles!(::InferenceChain, ::AbstractAttentionModel)
    error("not implemented")
end

function rejuvenate_attention!(chain::SeqPFChain,
                               attention::AbstractAttentionModel)
    # generate goal-driven hypotheses (ie. sensitivity)
    @time hypothesize!(chain, attention)
    # process those hypotheses into categorical weights for adaptive cycles
    goal_relevance!(chain, attention)
    # obtain the total amount of effort to be expended
    budget_cycles!(chain, attention)
    # main loop going through rejuvenation
    @time perturb_state!(chain, attention)
    return nothing
end

include("objectives.jl")
include("distances.jl")
include("uniform.jl")
include("sensitivity.jl")
include("perturb_state.jl")
