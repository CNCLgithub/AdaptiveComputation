
function get_n_scenes(dataset_path::String)
	file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    return n_scenes
end

"""
loads gt causal graphs and motion
"""
function load_scene(scene, dataset_path, gm_params)
	file = jldopen(dataset_path, "r")
    scene = read(file, "$scene")
    motion = scene["motion"]
    gt_causal_graphs = scene["gt_causal_graphs"]
    close(file)
    
    println("scene data loaded")

    scene_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :motion => motion])
end


export load_scene, get_n_scenes
