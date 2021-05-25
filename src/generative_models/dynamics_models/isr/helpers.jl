# MAYBE NO NEED FOR ANY OF THIS

using LinearAlgebra: cross


function poly_step_args(dm::SquishyDynamicsModel,
                        cg::CausalGraph)
    plys = @> cg begin
        filter_vertices((g, v) -> get_prop(cg, v, :object) isa Polygon)
        collect
    end

    np = length(plys)
    (fill(dm, np), fill(cg, np), plys)
end

function vert_step_args(dm::SquishyDynamicsModel,
                        cg::CausalGraph,
                        p::Polygon)
    (fill(dm, nv(p)), fill(cg, nv(p)), fill(p, nv(p)))
end

function update(dm::SquishyDynamicsModel,
                dot::Dot, rep::Vector{Float64}, dv::Vector{Float64})
    vel = dv + rep
    #vel *= dm.vel/norm(vel)
    x,y = dot.pos[1:2] + vel
    Dot([x,y,dot.pos[3]], vel)
end

function update(dm::SquishyDynamicsModel, obj::UGon, rep::Vector{Float64},
                dv::Vector{Float64}, dav::Float64)
    vel = obj.vel*dm.pol_inertia + dv + rep
    vel *= dm.vel/norm(vel)
    x,y = obj.pos[1:2] + vel
    UGon([x,y,obj.pos[3]], vel)
end

function update(dm::SquishyDynamicsModel,
                obj::NGon, rep::Vector{Float64},
                dv::Vector{Float64}, dav::Float64)
    vel = obj.vel*dm.pol_inertia + dv + rep
    vel *= dm.vel/norm(vel)
    x,y = obj.pos[1:2] + vel
    avel = obj.ang_vel*dm.pol_ang_inertia + dav
    rot = obj.rot + avel
    NGon([x,y,obj.pos[3]], rot, vel, avel,
            obj.radius, obj.nv)
end

function cross2d(a::Vector{Float64}, b::Vector{Float64})
    a1, a2 = a
    b1, b2 = b
    a1*b2-a2*b1
end

function vector_to(w::Wall, o::Object)
    # w.n/norm(w.n) .* dot(o.pos[1:2] - w.p1, w.n)
    # from https://stackoverflow.com/a/48137604
    @unpack p1, p2, n = w
    p3 = o.pos[1:2]
    -n .* cross2d(p2-p1,p3-p1) ./ norm(p2-p1)
end

function vector_to(a::Object, b::Object)
    b.pos[1:2] - a.pos[1:2]
end

function attraction(dm::SquishyDynamicsModel, pol::NGon, dot::Dot, order::Int64)
    @unpack poly_att_m, poly_att_a, poly_att_x0 = dm

    # find the position defined by the polygon and attract to it
    r = pol.radius
    c_dot_x = pol.pos[1] + r * cos(2*pi*order/nv(pol) + pol.rot)
    c_dot_y = pol.pos[2] + r * sin(2*pi*order/nv(pol) + pol.rot)

    vec = [c_dot_x, c_dot_y] - dot.pos[1:2] # vector
    d = norm(vec) # distance
    iszero(d) && return zeros(2)
    # f = poly_att_m * exp(poly_att_a * d - poly_att_x0) .* vec ./ d
    f = vec .* poly_att_m ./ (1.0 + exp(-poly_att_a*(d - poly_att_x0)))

    return f
end


function attraction(dm::SquishyDynamicsModel, pol::UGon, dot::Dot, order::Int64)
    @unpack poly_att_m, poly_att_a, poly_att_x0 = dm

    # find the position defined by the polygon and attract to it
    c_dot_x, c_dot_y = pol.pos[1:2]
    vec = [c_dot_x, c_dot_y] - dot.pos[1:2] # vector
    d = norm(vec) # distance
    iszero(d) && return zeros(2)
    # f = poly_att_m * exp(poly_att_a * d - poly_att_x0) .* vec ./ d
    f = vec .* poly_att_m ./ (1.0 + exp(-poly_att_a*(d - poly_att_x0)))
    return f
end


function repulsion(dm::SquishyDynamicsModel, a::Wall, b::Dot)
    vec = vector_to(a, b)
    d = norm(vec)
    uvec = vec / d
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    f = wall_rep_m * exp(-1 * (wall_rep_a * (d - wall_rep_x0))) .* uvec
    return f
end

function repulsion(dm::SquishyDynamicsModel, a::Wall, b::Polygon)
    vec = vector_to(a, b)
    r = radius(b)
    d = norm(vec)
    uvec = vec / d
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    f = wall_rep_m * exp(-1 * (wall_rep_a * (d - r - wall_rep_x0))) .* uvec
    return f
end

function repulsion(dm::SquishyDynamicsModel, a::Dot, b::Dot)
    vec = vector_to(a, b)
    d = norm(vec)
    uvec = vec/d
    @unpack vert_rep_m, vert_rep_a, vert_rep_x0 = dm
    f = vert_rep_m * exp(-1 * (vert_rep_a * (d - vert_rep_x0))) .* uvec
    return f
end

function repulsion(dm::SquishyDynamicsModel, a::Polygon, b::Polygon)
    vec = vector_to(a, b)
    d = norm(vec)
    uvec = vec/d
    @unpack poly_rep_m, poly_rep_a, poly_rep_x0 = dm
    f = poly_rep_m * exp(-1 * (poly_rep_a * (d - poly_rep_x0))) .* uvec
    return f
end

function get_walls(cg::CausalGraph, dm::SquishyDynamicsModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end

