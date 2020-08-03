export AbstractExperiment,
        run_inference,
        get_name

abstract type AbstractExperiment end

function run_inference(exp::AbstractExperiment, path::String)
    error("Not implemented")
end

function get_name(exp::AbstractExperiment)
    error("Not implemented")
end

include("example/example.jl")
include("sensitivity_td/sensitivity_td.jl")

# exp0
include("exp0/generic/exp0_generic.jl")
include("exp0/mask_rcnn/exp0_mask_rcnn.jl")

include("exp0/attention/exp0_attention.jl")
include("exp0/trial_avg/exp0_trial_avg.jl")
include("exp0/base/exp0_base.jl")

include("exp0/sensitivity_td/exp0_sensitivity_td.jl")
include("exp0/sensitivity_dc/exp0_sensitivity_dc.jl")
# include("exp0/exp0.jl")
