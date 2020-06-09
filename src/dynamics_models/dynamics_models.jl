abstract type DynamicsModel end

function update!(::DynamicsModel, objects::Vector{Object})
    error("not implemented")
end

include("brownian_dynamics_model.jl")

