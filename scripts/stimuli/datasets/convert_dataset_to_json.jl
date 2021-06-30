using MOT
using JSON

function convert_dataset_to_json(dataset_path, json_path; hgm=false)
    n_scenes = MOT.get_n_scenes(dataset_path)

    data = []

    for i=1:n_scenes
        scene_data = load_scene(i, dataset_path)

        cgs = scene_data[:gt_causal_graphs]
        positions = MOT.@>> cgs begin
            map(cg -> MOT.get_objects(cg, Dot))
            map(os -> map(o -> MOT.get_pos(o), os))
        end
        scene = Dict()
        scene[:positions] = positions
        scene[:aux_data] = scene_data[:aux_data]
        push!(data, scene)
    end

    open(json_path, "w") do f
        write(f, json(data))
    end
end
