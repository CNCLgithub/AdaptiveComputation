export load_scene

import Base.read

function try_read(scene, element::String)
    datum = nothing
    try
        datum = scene[element]
    catch
    end
    return datum
end

"""
    loads gt_causal_graphs and aux_data
"""
function load_scene(scene, dataset_path)
    
    file = jldopen(dataset_path, "r")
    scene = base.read(file, "$scene")
    gt_causal_graphs = scene["gt_causal_graphs"]
    aux_data = try_read(scene, "aux_data")
    close(file)
    println("scene data loaded")

    scene_data = dict([:gt_causal_graphs => gt_causal_graphs,
                       :aux_data => aux_data])
end
