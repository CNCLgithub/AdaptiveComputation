
@with_kw struct InertiaModel <: AbstractDynamicsModel
    vel::Float64 = 10 # base vel
    low_w::Float64 = 0.05
    high_w::Float64 = 3.5
    a::Float64 = 0.1
    b::Float64 = 0.3
end

function load(::Type{InertiaModel}, path::String)
    InertiaModel(;read_json(path)...)
end

include("gen.jl")
include("helpers.jl")

export InertiaModel
