using MOT
using JSON

function convert_dataset_to_json(dataset_path, json_path)
    n_scenes = MOT.get_n_scenes(dataset_path)

    data = []

    for i=1:n_scenes
        scene_data = load_scene(i, dataset_path, default_gm_params)
        cgs = scene_data[:gt_causal_graphs]
        positions = map(cg-> map(dot->dot.pos, cg.elements), cgs)
        push!(data, positions)
    end

    open(json_path, "w") do f
        write(f, json(data))
    end
end
