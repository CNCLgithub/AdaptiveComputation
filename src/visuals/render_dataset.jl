using JLD2

function render_dataset(dataset_path, render_path, gm_params;
                        freeze_time = 24,
                        stimuli = true,
                        highlighted_start = collect(1:gm_params.n_trackers),
                        highlighted_end = [1],
                        highlighted_start_color = "blue",
                        highlighted_end_color = "red")

    ispath(render_path) || mkpath(render_path)

    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    
    for i=1:n_scenes
        scene_data = MOT.load_scene(i, dataset_path, gm_params)
        path = joinpath(render_path, "$i")
        gt_cgs = scene_data[:gt_causal_graphs]
        T = length(gt_cgs) - 1
        MOT.render(gm_params, T;
                   gt_causal_graphs=gt_cgs,
                   freeze_time=freeze_time,
                   path=path,
                   stimuli=stimuli,
                   highlighted_start=highlighted_start,
                   highlighted_end=highlighted_end,
                   highlighted_start_color=highlighted_start_color,
                   highlighted_end_color=highlighted_end_color)

    end
end
