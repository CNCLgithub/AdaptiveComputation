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

include("exp0/exp0.jl")
