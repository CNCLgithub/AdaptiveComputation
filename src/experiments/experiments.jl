export AbstractExperiment

abstract type AbstractExperiment end

function run_inference(exp::AbstractExperiment)
    error("Not implemented")
end

include("example.jl")
