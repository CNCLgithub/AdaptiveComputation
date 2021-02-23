
function poly_step_args(dm::SquishyDynamicsModel,
                        cg::CausalGraph)
    plys = @> cg begin
        filter_vertices((g, v) -> get_prop(cg, v, :object) isa Polygon)
        collect
    end

    #plys = @>> vs map(v -> get_prop(cg, v, :object))
    np = length(plys)
    (fill(dm, np), fill(cg, np), plys)
end

function vert_step_args(dm::SquishyDynamicsModel,
                        cg::CausalGraph,
                        p::Polygon)
    (fill(dm, nv(p)), fill(cg, nv(p)), fill(p, nv(p)))
end

function update(dot::Dot, rep::Vector{Float64}, dv::Vector{Float64})
    vel = dot.vel + dv + rep
    x,y = dot.pos[1:2] + vel
    Dot([x,y,dot.pos[3]], vel)
end

function update(obj::UGon, rep::Vector{Float64},
                dv::Vector{Float64}, dav::Float64)
    vel = obj.vel + dv + rep
    x,y = obj.pos[1:2] + vel
    UGon([x,y,obj.pos[3]], vel)
end

function update(obj::NGon, rep::Vector{Float64},
                dv::Vector{Float64}, dav::Float64)
    vel = obj.vel + dv + rep
    x,y = obj.pos[1:2] + vel
    avel = obj.ang_vel * dav
    rot = obj.rot + avel
    NGon([x,y,obj.pos[3]], rot, vel, avel,
            obj.radius, obj.nv)
end

function distance_to(w::Wall, o::Object)
    dot(o.pos[1:2] - w.p1, w.n)
end

function distance_to(a::Object, b::Object)
    norm(a.pos-b.pos)
end

function attraction(dm::SquishyDynamicsModel, pol::NGon, dot::Dot, order::Int64)
    @unpack poly_att_m, poly_att_a, poly_att_x0 = dm

    # find the position defined by the polygon and attract to it
    r = pol.radius
    c_dot_x = pol.pos[1] + r * cos(2*pi*order/nv(pol) + pol.rot)
    c_dot_y = pol.pos[2] + r * sin(2*pi*order/nv(pol) + pol.rot)

    vec = [c_dot_x, c_dot_y] - dot.pos[1:2] # vector
    d = norm(vec) # distance
    f = poly_att_m * exp(poly_att_a * d - poly_att_x0) # magnitude of the force
    return vec/d .* f
end

# TODO unused
function repulsion(dm::SquishyDynamicsModel, a::Polygon, b::Dot)
    @unpack poly_rep_m, poly_rep_a, poly_rep_x0 = dm
    d = distance_to(a, b)
    r = norm(d)
    ud = d / nd
    theta = atan(ud[2], ud[1])
    # centripetal vel
    cv = -r * a.ang_vel*sin(theta) + r * a.ang_vel * cos(theta)
    # radial equilibrium
    # perhaps add segment equilibrium
    rv = (a.radius - r)
    rv = rv * poly_rep_m * exp(-1 * (poly_rep_a * nd - poly_rep_x0))
    rv = rv .* ud
    cv + rv
end

function repulsion(dm::SquishyDynamicsModel, a::Wall, b::Object)
    d = distance_to(a, b)
    nd = norm(d)
    ud = d / nd
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    wall_rep_m * exp(-1 * (wall_rep_a * nd - wall_rep_x0)) .* ud
end

function repulsion(dm::SquishyDynamicsModel, a::Dot, b::Dot)
    d = distance_to(a, b)
    nd = norm(d)
    ud = d ./ nd
    @unpack vert_rep_m, vert_rep_a, vert_rep_x0 = dm
    vert_rep_m * exp(-1 * (vert_rep_a * nd - vert_rep_x0)) .* ud
end

function get_walls(cg::CausalGraph, dm::SquishyDynamicsModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end

"""
Takes a list of `Tuple{Polygon, Dot[]}` and creates a new `CausalGraph`
"""
function process_temp_state(current_state, old_cg::CausalGraph, dm::SquishyDynamicsModel)

    cg = CausalGraph(SimpleDiGraph())

    # getting the walls from the previous causal graph
    ws = get_walls(old_cg, dm) 
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end

    set_prop!(cg, :walls, walls_idx(dm))
    for (poly, verts) in current_state
        add_vertex!(cg)
        poly_v = MetaGraphs.nv(cg)
        set_prop!(cg, poly_v, :object, poly)

        for (index, v) in enumerate(verts)
            add_vertex!(cg)
            vi = MetaGraphs.nv(cg)
            add_edge!(cg, poly_v, vi)
            set_prop!(cg, vi, :object, v)
            # to calculate position wrt polygon
            set_prop!(cg, vi, :order, index) 
            set_prop!(cg, Edge(poly_v, vi),
                      :parent, true)
        end
    end

    calculate_repulsion!(cg, dm)
    return cg
end

walls_idx(dm::SquishyDynamicsModel) = collect(1:4)
