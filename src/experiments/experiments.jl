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
include("exp0/generic/exp0_generic.jl")
include("exp0/attention/exp0_attention.jl")
include("sensitivity_td/sensitivity_td.jl")
