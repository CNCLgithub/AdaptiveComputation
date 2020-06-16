export AbstractDynamicsModel

abstract type AbstractDynamicsModel end

# function update(::AbstractDynamicsModel)
#     error("not implemented")
# end

include("brownian_dynamics_model.jl")

