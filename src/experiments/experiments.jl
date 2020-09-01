export AbstractExperiment,
        generate_inference_visualize,
        run_inference,
        get_name

abstract type AbstractExperiment end

function run_inference(exp::AbstractExperiment,
                       gm::GMMaskParams,
                       motion::AbstractDynamicsModel,
                       init_positions,
                       masks,
                       attention::T,
                       path::String) where {T<:AbstractAttentionModel}
    error("Not implemented")
end

function get_name(exp::AbstractExperiment)
    error("Not implemented")
end

include("exp0/exp0.jl")
include("exp1/exp1.jl")
include("exp1_isr/exp1_isr.jl")

# ISR dynamics
include("isr_dynamics/isr_dynamics.jl")

