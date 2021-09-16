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
function dynamics_update!(cg::CausalGraph, dm::InertiaModel)
    for v in LightGraphs.vertices(cg)
        obj = get_prop(cg, v, :object)
        dynamics_update!(cg, dm, v, obj)
    end
    return nothing
end


"""
Catchall for undefined dynamics
"""
function dynamics_update!(cg::CausalGraph, dm::InertiaModel,
                          v::Int64, obj::Thing)
    return nothing
end

"""
Vertex update logic.
First update vertices given walls and accumulate interactions
"""
function dynamics_update!(cg::CausalGraph, dm::InertiaModel,
                          v::Int64, obj::Dot)

    # update the interaction of wall -> x
    @>> cg begin
        walls
        foreach(w -> dynamics_update!(cg, dm, w, v))
    end

    # update v -> v?
    # TODO?

    # accumulate forces
    f = zeros(2)
    for i in inneighbors(cg, v)
        f += get_prop(cg, Edge(i, v), :force)
    end
    set_prop!(cg, v, :object,
              update(dm, obj, f))
    return nothing
end

"""
Define vertex to vertex interactions as edges
"""
function dynamics_update!(cg::CausalGraph, dm::InertiaModel,
                          w::Int64, v::Int64)
    # dont assign edges to self
    w === v && return nothing
    a = get_prop(cg, w, :object)
    b = get_prop(cg, v, :object)
    add_edge!(cg, w, v)
    set_prop!(cg, Edge(w, v),
              :force, force(dm, a, b))
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
    f = wall_rep_m * exp(-1 * (wall_rep_a * (d - wall_rep_x0))) .* uvec
    # @unpack wall_repulsion, distance = dm
    # f = wall_repulsion*exp(-(v[1]^2 + v[2]^2)/(distance^2))
    # force .+= absolute_force * v/norm(v)
    return f
end

function update(dm::InertiaModel,
                dot::Dot, rep::Vector{Float64})
    @unpack pos, vel = dot
    # undo vel step
    pos[1:2] -= vel

    # change direction without changing magnitude
    m = norm(vel)
    vel += rep
    m /= norm(vel)
    if !(isinf(m) || isnan(m))
        vel *= m
    end

    pos[1:2] += vel
    # @show vel
    # @show pos
    Dot(pos = pos, vel = vel, radius = dot.radius)
end
