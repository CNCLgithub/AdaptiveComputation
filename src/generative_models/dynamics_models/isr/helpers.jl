

function process_temp_state(current_state, old_cg::CausalGraph, dm::ISRDynamics)
    cg = CausalGraph(SimpleDiGraph())
    
    for tracker in current_state
        add_vertex!(cg)
        tracker_v = MetaGraphs.nv(cg)
        set_prop!(cg, tracker_v, :object, tracker)
    end

    calculate_repulsion!(cg, dm)
    return cg
end


function process_temp_state(current_state, gm::GMParams, dm::ISRDynamics)
    cg = CausalGraph(SimpleDiGraph())
    process_temp_state(current_state, cg, dm)
end
