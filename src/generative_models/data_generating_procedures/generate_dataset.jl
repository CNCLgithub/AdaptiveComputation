export generate_dataset

function generate_dataset(dataset_path, n_trials, k, gm, motion)
    jldopen(dataset_path, "w") do file 
        file["n_trials"] = n_trials
        for i=1:n_trials
            init_positions, init_vels, masks, positions = dgp(k, gm, motion)

            trial = JLD2.Group(file, "$i")
            trial["gm"] = gm
            trial["motion"] = motion
            trial["positions"] = positions
            trial["init_positions"] = init_positions
        end
    end
end
