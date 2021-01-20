using JLD2

function render_dataset(dataset_path, render_path, gm_params)
    ispath(render_path) || mkpath(render_path)

    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    
    for i=1:n_scenes
        scene_data = MOT.load_scene(i, dataset_path, gm_params)
        path = joinpath(render_path, "$i")
        gt_cgs = scene_data[:gt_causal_graphs]
        k = length(gt_cgs) - 1
        MOT.render(gm_params, k;
                   gt_causal_graphs=gt_cgs,
                   freeze_time=24,
                   path=path,
                   stimuli=true,
                   highlighted=collect(1:gm_params.n_trackers))
    end
end
