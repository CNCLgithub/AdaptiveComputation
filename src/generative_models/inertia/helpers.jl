
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
