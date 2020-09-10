export load_trial

"""
loads gt causal graphs and motion
"""
function load_trial(trial, dataset_path, gm;
                    generate_masks=true,
                    from_mask_rcnn=false)
    
	file = jldopen(dataset_path, "r")
    trial = read(file, "$trial")
    motion = trial["motion"]
    gt_causal_graphs = trial["gt_causal_graphs"]
    close(file)

    if generate_masks
        masks = get_masks(gt_causal_graphs[2:end], gm)
        if from_mask_rcnn
            masks[1:k-1] = get_masks_from_mask_rcnn(gt_causal_graphs[2:end], gm)[1:k-1]
        end
    else
        masks = nothing
    end
    
    trial_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :motion => motion,
                       :masks => masks])
    return trial_data
end
