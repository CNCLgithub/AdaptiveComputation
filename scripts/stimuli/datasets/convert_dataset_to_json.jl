# using MOT
using Lazy: @>>
using MOT: get_objects, get_pos, Dot
using JLD2
using JSON

function repair_isr_480()
end

function convert_dataset_to_json(dataset_path, json_path; hgm=false)
    n_scenes = 60 #MOT.get_n_scenes(dataset_path)
    data = []

    jldopen(dataset_path, "r") do file
        for i=1:n_scenes
            d = file["$i"]
            display(d)
            display(d["motion"])
            cgs = d["gt_causal_graphs"]
            positions = @>> cgs begin
                map(cg -> get_objects(cg, Dot))
                map(os -> map(o -> get_pos(o), os))
            end
            scene = Dict()
            scene[:positions] = positions
            scene[:aux_data] = d[:aux_data]
            push!(data, scene)
        end
    end
    open(json_path, "w") do f
        write(f, json(data))
    end
end
