function dynamics_init!(dm::InertiaModel, cg::CausalGraph, things::AbstractArray{Any, 1})
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))

    cg = dynamics_update!(dm, cg, things)
    return cg
end

"""
Takes a list of `Tuple{Polygon, Dot[]}` and creates a new `CausalGraph`
"""
function dynamics_update!(dm::InertiaModel, prev_cg::CausalGraph,
                         vs::Vector{Int64}, things::AbstractArray{Any, 1})
    cg = deepcopy(prev_cg)

    for (i, thing) in enumerate(things)
        set_prop!(cg, vs[i], :object, thing)
    end

    return cg
end

walls_idx(dm::InertiaModel) = collect(1:4)


function get_walls(cg::CausalGraph, dm::InertiaModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end
