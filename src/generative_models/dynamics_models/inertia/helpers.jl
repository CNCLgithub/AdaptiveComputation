
function dynamics_init(dm::InertiaModel, gm::GMParams,
                       cg::CausalGraph, things)
    cg = deepcopy(cg)
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

function dynamics_update(dm::InertiaModel,
                         cg::CausalGraph,
                         things)
    cg = deepcopy(cg)
    vs = get_object_verts(cg, Dot)
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

function inertia_step_args(prev_cg::CausalGraph)
    cg = deepcopy(prev_cg)
    vs = get_object_verts(cg, Dot)
    cgs = fill(cg, length(vs))
    (cgs, vs)
end
