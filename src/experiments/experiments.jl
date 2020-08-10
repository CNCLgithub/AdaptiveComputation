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

# Exp0
include("exp0/exp0.jl")

# ISR dynamics
include("isr_dynamics/isr_dynamics.jl")
