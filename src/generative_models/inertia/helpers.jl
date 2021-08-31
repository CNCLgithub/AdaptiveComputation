
walls_idx(dm::InertiaModel) = collect(1:4)

function get_walls(cg::CausalGraph, dm::InertiaModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end

function inertia_step_args(cg::CausalGraph)
    vs = get_object_verts(cg, Dot)
    cgs = fill(cg, length(vs))
    (cgs, vs)
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
