
"""
High level update.
Any global changes happen here
"""
function dynamics_update(dm::InertiaModel, cg::CausalGraph)::Diff
    # `Wall`s don't change.
    # `UniformEnsemble` might change
    st = StaticPath[]
    ch = ChangeDict()

    # foreach(v -> (print("$(v) => "); display(props(cg, v))), LightGraphs.vertices(cg))
    for v in get_object_verts(cg, Union{Wall, Dot})
        obj = get_prop(cg, v, :object)
        isa(obj, Wall) && push!(st, v => :object)
        # Resolve forces all other elements
        isa(obj, Dot) && dynamics_update!(ch, dm, cg, v, obj)
    end
    Diff(Thing[], Int64[], st, ch)
end


"""
Catchall for undefined dynamics
"""
function dynamics_update!(ch::ChangeDict,
                          dm::InertiaModel,
                          cg::CausalGraph,
                          v::Int64,
                          obj::Thing)
    return nothing
end

"""
Vertex update logic.
First update vertices given walls and accumulate interactions
"""
function dynamics_update!(ch::ChangeDict,
                          dm::InertiaModel,
                          cg::CausalGraph,
                          v::Int64,
                          obj::Dot)

    # update the interaction of wall -> x
    @>> cg begin
        walls
        # computes the force of w on v
        foreach(w -> dynamics_update!(ch, dm, cg, w, v))
    end

    # update v -> v?
    # TODO?

    return nothing
end

"""
Define vertex to vertex interactions as edges
"""
function dynamics_update!(ch::ChangeDict,
                          dm::InertiaModel,
                          cg::CausalGraph,
                          w::Int64,
                          v::Int64)
    # dont assign edges to self
    w === v && return nothing
    a = get_prop(cg, w, :object)
    b = get_prop(cg, v, :object)
    ch[Edge(w, v) => :force] = force(dm, a, b)
    return nothing
end


# Current interaction rules:
# wall -> Dot


"""
Wall -> Dot
"""
function force(dm::InertiaModel, a::Wall, b::Dot)
    vec = vector_to(a, b)
    d = norm(vec)
    uvec = vec / d
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    wall_rep_m * exp(-1 * (wall_rep_a * (d - wall_rep_x0))) .* uvec
end
