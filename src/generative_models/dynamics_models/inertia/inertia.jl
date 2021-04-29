"""
@with_kw struct InertiaModel <: AbstractDynamicsModel
    vel::Float64 = 10 # base vel
    low_w::Float64 = 0.05
    high_w::Float64 = 3.5
    a::Float64 = 0.1
    b::Float64 = 0.3
end
"""

@with_kw struct InertiaModel <: AbstractDynamicsModel
    vel::Float64 = 10 # base vel
    a::Float64 = 0.1
    b::Float64 = 0.3
    k_min::Float64 = 0.5 # von_misses kappa for angle
    k::Float64 = 100.0 # von_misses kappa for angle
    w::Float64 = 2.5 # standard deviation for magnitude noise
end

function load(::Type{InertiaModel}, path::String)
    InertiaModel(;read_json(path)...)
end

include("gen_v2.jl")
include("helpers.jl")

export InertiaModel
