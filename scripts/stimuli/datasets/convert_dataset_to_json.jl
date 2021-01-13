using MOT
using JSON

function convert_dataset_to_json(dataset_path, json_path; hgm=false)
    n_scenes = MOT.get_n_scenes(dataset_path)

    data = []

    for i=1:n_scenes
        scene_data = load_scene(i, dataset_path, default_gm;
                                generate_masks=false)
        cgs = scene_data[:gt_causal_graphs]
        positions = nothing
        if hgm
        # if hierarchical, extract invidual dot positions
            n_dots = length(scene_data[:aux_data][:targets])
            positions = map(cg -> MOT.get_hgm_positions(cg, fill(true, n_dots)), cgs)
        else
        # else just extract positions of all the dots
            positions = map(cg-> map(dot->dot.pos, filter(x->isa(x, Dot), cg.elements)), cgs)
        end
        scene = Dict()
        scene[:positions] = positions
        if hgm
            scene[:aux_data] = scene_data[:aux_data]
        end
        push!(data, scene)
    end

    open(json_path, "w") do f
        write(f, json(data))
    end
end
