function dynamics_init!(cg::CausalGraph,
                        dm::InertiaModel,
                        gm::GMParams,
                        things::AbstractArray{Thing})

    ws = init_walls(gm.area_width, gm.area_height)
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))

    for thing in things
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        set_prop!(cg, v, :object, thing)
    end

    return cg
end

"""
High level update.
Any global changes happen here
"""
function dynamics_update(dm::InertiaModel, cg::CausalGraph)::Diff
    # `Wall`s don't change.
    # `UniformEnsemble` might change
    st = StaticPath[]
    for w in get_object_verts(cg, Wall)
        push!(st, w => :object)
    end

    # Resolve forces all other elements
    changed = Dict{ChangeDiff, Any}
    for v in LightGraphs.vertices(cg)
        obj = get_prop(cg, v, :object)
        dynamics_update!(changed, cg, dm, v, obj)
    end
    return nothing
end


"""
Catchall for undefined dynamics
"""
function dynamics_update!(ch::Dict,
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
function dynamics_update!(ch::Dict{ChangeDiff},
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
function dynamics_update!(ch::Dict{ChangeDiff},
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
