export SquishyDynamicsModel

@with_kw struct SquishyDynamicsModel <: AbstractDynamicsModel

    vel::Float64 = 10.0 # base velocity

    pol_inertia = 0.9
    pol_ang_inertia = 0.9
    pol_sigma = 0.5
    vert_sigma = 0.0
    pol_ang_vel_sigma = 0.01
    
    # poly att-> vert
    poly_att_m = 0.5
    poly_att_a = 0.05
    poly_att_x0 = 0.0
    
    # poly rep-> poly
    poly_rep_m = 1.0
    poly_rep_a = 0.005
    poly_rep_x0 = 0.0

    # wall rep-> *
    wall_rep_m = 10.0
    wall_rep_a = 0.02
    wall_rep_x0 = 0.0
    
    # vert rep-> vert
    vert_rep_m = 35.0
    vert_rep_a = 0.016
    vert_rep_x0 = 0.0
end

function load(::Type{SquishyDynamicsModel}, path::String)
    SquishyDynamicsModel(;read_json(path)...)
end


include("dynamics.jl")
include("gen.jl")
