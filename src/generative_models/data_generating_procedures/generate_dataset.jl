export generate_dataset

function generate_dataset(dataset_path, n_scenes, k, gm, motion)
    jldopen(dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            trial_data = dgp(k, gm, motion;
                             generate_masks=false)

            trial = JLD2.Group(file, "$i")
            trial["gm"] = gm
            trial["motion"] = motion
            trial["gt_causal_graphs"] = trial_data[:gt_causal_graphs]
        end
    end
end
