function get_n_scenes(dataset_path::String)
	file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    return n_scenes
end

include("get_masks_from_mask_rcnn.jl")
include("load_scene.jl")
include("load_exp0_scene.jl")
