
function get_walls(cg::CausalGraph, dm::InertiaModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end

"""
Takes a list of `Tuple{Polygon, Dot[]}` and creates a new `CausalGraph`
"""
function process_temp_state(current_state, old_cg::CausalGraph, dm::InertiaModel)

    cg = CausalGraph(SimpleDiGraph())

    # getting the walls from the previous causal graph
    ws = get_walls(old_cg, dm) 
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))
    
    for dot in current_state
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        set_prop!(cg, v, :object, dot)
    end

    #calculate_repulsion!(cg, dm)
    return cg
end

walls_idx(dm::InertiaModel) = collect(1:4)

function process_temp_state(current_state, gm::GMParams, dm::InertiaModel)

    cg = CausalGraph(SimpleDiGraph())

    # getting the walls from the previous causal graph
    ws = init_walls(gm)
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))
    process_temp_state(current_state, cg, dm)
end
