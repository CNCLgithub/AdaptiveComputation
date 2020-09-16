using MOT
using JLD2
using DataFrames
using CSV


"""
    Saves a csv at `out` directory with each scene's motion parameters
"""
function scene_data(dataset_path, out)
    @load dataset_path n_scenes

    # getting the motion parameters
    scene_data = load_scene(1, dataset_path, default_gm;
                            generate_masks=false)
    parameters = fieldnames(typeof(scene_data[:motion]))
    
    # initializing dataframe with parameters
    df = DataFrame(scene=1:n_scenes)
    map(p->df[!,p] .= 0.0, parameters) 

    for i=1:n_scenes
        scene_data = load_scene(i, dataset_path, default_gm;
                                generate_masks=false)
        motion = scene_data[:motion]
        for p in parameters
            df[i, p] = getfield(motion, p)
        end
    end
    mkpath(out)
    CSV.write(joinpath(out, "scene_data.csv"), df)
end
