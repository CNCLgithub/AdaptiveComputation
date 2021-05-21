export ISRDynamics

@with_kw struct ISRDynamics <: AbstractDynamicsModel
    repulsion::Bool = true
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance::Float64 = 60.0
    vel::Float64 = 10.0 # base velocity
    rep_inertia::Float64 = 0.9

    brownian::Bool = true
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_x::Float64 = 1.0
    sigma_y::Float64 = 1.0
end

function load(::Type{ISRDynamics}, path::String)
    ISRDynamics(;read_json(path)...)
end

