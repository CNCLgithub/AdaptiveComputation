export InertiaModel

################################################################################
# Model definition
################################################################################

"""
Model that uses inertial change points to "explain" interactions
"""
@with_kw struct InertiaModel <: AbstractDynamicsModel

    # stochastic motion parameters
    vel::Float64 = 10 # base vel
    bern::Float64 = 0.9
    k_min::Float64 = 0.5 # min von_misses kappa for angle
    k_max::Float64 = 100.0 # max von_misses kappa for angle
    w_min::Float64 = 2.5 # min standard deviation for magnitude noise
    w_max::Float64 = 5.5 # max standard deviation for magnitude noise

    # force parameters
    # wall rep-> *
    wall_rep_m::Float64 = 0.0
    wall_rep_a::Float64 = 0.02
    wall_rep_x0::Float64 = 0.0
end

function load(::Type{InertiaModel}, path::String)
    InertiaModel(;read_json(path)...)
end

include("helpers.jl")
include("dynamics.jl")
include("kinematics.jl")
include("causation.jl")
include("gen.jl")
