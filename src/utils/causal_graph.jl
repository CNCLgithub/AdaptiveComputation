using LightGraphs
using MetaGraphs
const CausalGraph = MetaGraphs.MetaDiGraph{Int64, Float64}

get_dm(cg::CausalGraph) = get_prop(cg, :dm)
get_gm(cg::CausalGraph) = get_prop(cg, :gm)
get_graphics(cg::CausalGraph) = get_prop(cg, :graphics)

#function init_cg(gm::AbstractGMParams, dm::AbstractDynamicsModel, graphics::AbstractGraphics)
# idk how to properly solve this interdependence conflict CausalGraph <-> params
function get_init_cg(gm, dm, graphics)

    cg = CausalGraph(SimpleDiGraph())

    set_prop!(cg, :gm, gm)
    set_prop!(cg, :dm, dm)
    set_prop!(cg, :graphics, graphics)

    return cg
end

# if generating just the motion
function get_init_cg(gm, dm)

    cg = CausalGraph(SimpleDiGraph())

    set_prop!(cg, :gm, gm)
    set_prop!(cg, :dm, dm)

    return cg
end


# assuming first N vertices are walls
walls(cg::CausalGraph) = get_prop(cg, :walls)

function force(cg::CausalGraph, v::Int64)
    fs = @>> v begin
        inneighbors(cg)
        map(i -> Edge(i, v))
        Base.filter(e -> has_prop(cg, e, :force))
        map(e -> get_prop(cg, e, :force))
    end

    return isempty(fs) ? zeros(2) : sum(fs)
end

LightGraphs.vertices(cg::CausalGraph, v::Int64) = @>> v begin
    outneighbors(cg)
    collect(Int64)
    Base.filter(i -> has_prop(cg, Edge(v, i), :parent))
end


parent(cg::CausalGraph, v::Int64) = @>> v begin
    inneighbors(cg)
    Base.filter(i -> has_prop(cg, Edge(i, v), :parent))
    first
    #(i -> get_prop(cg, i, :object))
end

get_objects(cg::CausalGraph, type::Type) = @>> cg begin
    vertices
    map(v -> get_prop(cg, v, :object))
    Base.filter(v -> v isa type)
end

function get_object_verts(cg::CausalGraph, type::Type)
    vs = filter_vertices(cg, (g, v) -> get_prop(g, v, :object) isa type)
    collect(vs)
end
