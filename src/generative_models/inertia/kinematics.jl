"""
High level update.
Any global changes happen here
"""
function kinematics_update(dm::InertiaModel, cg::CausalGraph)::Diff

    # The only thing that changes are `Dot` kinematics
    # ie. forces are resolved to update position and vel

    # `Diff` implicitly treats everything as static when
    # static is empty.
    st = StaticPath[]
    ch = ChangeDict()
    for v in LightGraphs.vertices(cg)
        obj = get_prop(cg, v, :object)
        kinematics_update!(ch, dm, cg, v, obj)
    end
    Diff(Thing[], Int64[], st, ch)
end



"""
Catchall for undefined kinematics
"""
function kinematics_update!(ch::ChangeDict,
                            dm::InertiaModel,
                            cg::CausalGraph,
                            v::Int64,
                            obj::Thing)
    return nothing
end

function kinematics_update!(ch::ChangeDict,
                            dm::InertiaModel,
                            cg::CausalGraph,
                            v::Int64,
                            obj::Dot)
    # accumulate forces
    f = zeros(2)
    for i in inneighbors(cg, v)
        f += get_prop(cg, Edge(i, v), :force)
    end
    ch[v => :object] = update(dm, obj, f)
    return nothing
end

function update(dm::InertiaModel,
                dot::Dot,
                rep::Vector{Float64})
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
    Dot(pos = pos, vel = vel, radius = dot.radius)
end
