export SquishyDynamicsModel

@with_kw struct SquishyDynamicsModel <: AbstractDynamicsModel

    # repulsion
    dot_repulsion::Float64 = 10.0
    wall_repulsion::Float64 = 10.0
    distance::Float64 = 100.0
    vel::Float64 = 10.0 # base velocity

end

function load(::Type{SquishyDynamicsModel}, path::String)
    SquishyDynamicsModel(;read_json(path)...)
end



"""
Current repulsion rules:

wall -> *
vert -> vert
"""
function calculate_repulsion!(cg::CausalGraph, v::Int64, dm::SquishyDynamicsModel)
    obj = get_prop(cg, v)
    calculate_repulsion!(cg, dm, v, obj)
end

function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              v::Int64, obj::Polygon)
    @>> cg begin
        walls
        foreach(w -> calculate_repulsion!(cg, w, v))
    end
    return nothing
end

function calculate_repulsion!(cg::CausalGraph, dm::SquishyDynamicsModel,
                              v::Int64, obj::Dot)
    @>> cg begin
        walls
        foreach(w -> calculate_repulsion!(cg, w, v))
    end

    c = parent(cg, v)
    calculate_repulsion!(cg, c, v)

    @>> cg begin
        vertices
        Base.filter(i -> get_prop(cg, i, :object) isa Dot)
        foreach(i -> calculate_repulsion!(cg, i, v))
    end
    return nothing
end

function calculate_repulsion!(cg::CausalGraph, w::Int64, v::Int64)
    a = get_prop(cg, w, :object)
    b = get_prop(cg, v, :object)
    add_edge!(cg, w, v)
    set_prop!(cg, Edge(w, v),
              :force, repulsion(a, b))
    return nothing
end

include("helpers.jl")
include("gen.jl")
