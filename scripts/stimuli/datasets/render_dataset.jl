using MOT
using JLD2

function render_dataset(dataset_path, render_path;
                        freeze_time = 24)
    ispath(render_path) || mkpath(render_path)

    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    
    for i=1:n_scenes
        path = joinpath(render_path, "$i")

        scene_data = MOT.load_scene(i, dataset_path, default_gm;
                                    generate_masks=false)
        #gm = scene_data[:gm]
        gt_cgs = scene_data[:gt_causal_graphs]
    
        targets = nothing
        try
            targets = scene_data[:aux_data][:targets]
        catch
        end

        highlighted = isnothing(targets) ? collect(1:default_gm.n_trackers) : collect(1:length(targets))[targets]

        k = length(gt_cgs) - 1
        MOT.render(default_gm, k;
                   gt_causal_graphs=gt_cgs,
                   freeze_time=freeze_time,
                   path=path,
                   stimuli=true,
                   highlighted=highlighted)
    end
end
