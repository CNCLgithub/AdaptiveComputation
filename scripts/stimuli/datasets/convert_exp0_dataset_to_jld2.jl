using MOT
using JLD2

function convert_dataset(old_dataset_path, new_dataset_path,
                         n_scenes, gm)
    jldopen(new_dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            trial_data = load_exp0_trial(i, gm, old_dataset_path;
                                         generate_masks=false)
            cgs = trial_data[:gt_causal_graphs]

            gt_causal_graphs = Vector{CausalGraph}(undef, length(cgs) + 1)
            gt_causal_graphs[1] = cgs[1]
            gt_causal_graphs[2:end] = cgs

            trial = JLD2.Group(file, "$i")
            trial["gm"] = gm
            trial["motion"] = trial_data[:motion]
            trial["gt_causal_graphs"] = gt_causal_graphs
        end
    end
end

gm = GMMaskParams(exp0=true)

convert_dataset("/datasets/exp_0.h5",
                "/datasets/exp0.jld2",
                128,
                gm)
