using MOT
using JLD2

function render_dataset(dataset_path, render_path;
                        freeze_time = 24,
                        scenes = []) # empty means all
    ispath(render_path) || mkpath(render_path)

    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    
    scenes = isempty(scenes) ? collect(1:n_scenes) : scenes
    
    for i in scenes
        path = joinpath(render_path, "$i")

        scene_data = MOT.load_scene(i, dataset_path, default_gm;
                                    generate_masks=false)
        gm = scene_data[:gm]
        gt_cgs = scene_data[:gt_causal_graphs]
    
        targets = nothing
        try
            targets = scene_data[:aux_data][:targets]
        catch
        end

        println(scene_data[:aux_data])
        
        k = length(gt_cgs) - 1

        println(targets)
        
        MOT.render(gm, k;
                   gt_causal_graphs=gt_cgs,
                   freeze_time=freeze_time,
                   path=path,
                   highlighted_start=targets)
    end
end
