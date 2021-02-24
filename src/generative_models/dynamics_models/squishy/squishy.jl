export SquishyDynamicsModel

@with_kw struct SquishyDynamicsModel <: AbstractDynamicsModel

    # repulsion
    dot_repulsion::Float64 = 10.0
    wall_repulsion::Float64 = 10.0
    distance::Float64 = 100.0
    vel::Float64 = 10.0 # base velocity

    pol_inertia = 0.0
    pol_sigma = 0.02
    vert_sigma = 0.02
    pol_ang_vel_sigma = 0.0

    poly_att_m = 0.2
    poly_att_a = 0.001
    poly_att_x0 = -10

    wall_rep_m = 0.0001
    wall_rep_a = 0.01
    wall_rep_x0 = 9.0

    vert_rep_m = 0.002
    vert_rep_a = 0.05
    vert_rep_x0 = 8.5
end

function load(::Type{SquishyDynamicsModel}, path::String)
    SquishyDynamicsModel(;read_json(path)...)
end



"""
Current repulsion rules:

wall -> *
vert -> vert
"""


function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              v::Int64, obj::Object)
    return nothing
end

function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              v::Int64, obj::Polygon)
    @>> cg begin
        walls
        foreach(w -> calculate_repulsion!(cg, dm, w, v))
    end
    return nothing
end

function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              v::Int64, obj::Dot)
    @>> cg begin
        walls
        foreach(w -> calculate_repulsion!(cg, dm, w, v))
    end

    c = parent(cg, v)
    calculate_attraction!(cg, dm, c, v)

    @>> LightGraphs.vertices(cg) begin
        Base.filter(i -> get_prop(cg, i, :object) isa Dot)
        foreach(i -> calculate_repulsion!(cg, dm, i, v))
    end
    return nothing
end

function calculate_attraction!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              p::Int64, d::Int64)
    pol = get_prop(cg, p, :object)
    dot = get_prop(cg, d, :object)
    order = get_prop(cg, d, :order)
    add_edge!(cg, p, d)
    set_prop!(cg, Edge(p, d),
              :force, attraction(dm, pol, dot, order))
    return nothing
end

function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              w::Int64, v::Int64)
    # dont assign edges to self
    w == v && return nothing
    a = get_prop(cg, w, :object)
    b = get_prop(cg, v, :object)
    add_edge!(cg, w, v)
    set_prop!(cg, Edge(w, v),
              :force, repulsion(dm, a, b))
    return nothing
end

function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel)
    for v in LightGraphs.vertices(cg)
        obj = get_prop(cg, v, :object)
        calculate_repulsion!(cg, dm, v, obj)
    end
    return nothing
end

include("helpers.jl")
include("gen.jl")
