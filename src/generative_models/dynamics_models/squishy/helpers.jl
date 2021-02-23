
function poly_step_args(dm::SquishyDynamicsModel,
                        cg::CausalGraph,
                        p::Polygon)
    plys = @>> cg begin
        vertices
        Base.filter(v -> get_prop(cg, v, :object) isa Polygon)
    end
    np = length(plys)
    (fill(dm, np), fill(cg, np), plys)
end

function vert_step_args(dm::SquishyDynamicsModel,
                        cg::CausalGraph,
                        p::Polygon)
    (fill(dm, p.nv), fill(cg, p.nv), fill(p, p.nv))
end

function update(obj::Dot, vel::Vector{Float64})
    x,y = dot.pos[1:2] + vel
    Dot([x,y,dot.pos[3]], vel)
end

function update(obj::Polygon, rep::Vector{Float64},
                dv::Vector{Float64}, dva::Float64)
    vel = obj.vel * dv + rep
    x,y = obj.pos[1:2] + vel
    avel = obj.ang_vel * dav
    rot = obj.rot + avel
    Polygon([x,y,obj.pos[3]], vel, rot, avel,
            obj.radius, object.nv)
end


function repulsion(dm::SquishyDynamicsModel, a::Polygon, b::Dot)
    d = distance_to(a, b)
    r = norm(d)
    ud = d / nd
    theta = atan(ud[2], ud[1])
    # centripetal vel
    cv = -r * a.ang_vel*sin(theta) + r * a.ang_vel * cos(theta)
    # radial equilibrium
    # perhaps add segment equilibrium
    @unpack poly_rep_m, poly_rep_a, poly_rep_x0 = dm
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

"""
Takes a list of `Tuple{Polygon, Dot[]}` and creates a new `CausalGraph`
"""
function process_temp_state(dm::SquishyDynamicsModel, current_state)

    cg = CausalGraph(SimpleDiGraph())

    for w in walls(dm)
        add_vertex!(cg)
        set_props!(cg, w, :object, w)
    end

    # TODO add walls
    set_prop!(cg, :walls, ws)
    for (poly, verts) in current_state
        add_vertex!(cg)
        poly_v = nv(cg)
        set_prop!(cg, poly_v, :object, poly)

        for v in verts
            add_vertex!(cg)
            vi = nv(cg)
            add_edge!(cg, poly_v, vi)
            set_props!(cg, vi, :object, v)
            set_prop!(cg, Edge(poly_v, vi),
                      :parent, true)
        end
    end

    calculate_repulsion!(cg, gm)
    return cg
end
