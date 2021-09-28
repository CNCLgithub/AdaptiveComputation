export load_scene

# Assumes a fixed number of objects per time point
# and that all objects are `Dot`s.
#
# Also assumes that all other properties other than
# position are irrelevant.
function cg_from_positions(positions)
    nt = length(positions)
    cgs = Vector{CausalGraph}(undef, nt)
    for t = 1:nt
        g = CausalGraph()
        step_pos = positions[t]
        for j = 1:length(step_pos)
            d = Dot(pos = step_pos[j])
            add_vertex!(g)
            set_prop!(g, j, :object, d)
        end
        cgs[t] = g
    end
    cgs
end

"""
    loads gt_causal_graphs and aux_data
"""
function load_scene(dataset_path::String, scene::Int64)
    scene_data = JSON.parsefile(dataset_path)[scene]
    aux_data = scene_data["aux_data"]
    cgs = cg_from_positions(scene_data["positions"])
    scene_data = Dict(:gt_causal_graphs => cgs,
                       :aux_data => aux_data)
end
