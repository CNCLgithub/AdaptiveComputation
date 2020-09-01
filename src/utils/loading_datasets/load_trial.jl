export load_trial

"""
loads init_positions, masks and the motion model from dataset
"""
function load_trial(trial, dataset_path, gm;
                    generate_masks=true,
                    from_mask_rcnn=false)
    
	file = jldopen(dataset_path, "r")
    trial = read(file, "$trial")
    motion = trial["motion"]
    positions = trial["positions"]
    init_positions = trial["init_positions"]
    close(file)

    if generate_masks
        masks = get_masks(positions,
                          gm.dot_radius,
                          gm.img_height, gm.img_width,
                          gm.area_height, gm.area_width)
        if from_mask_rcnn
            masks[1:k-1] = get_masks_from_mask_rcnn(positions, gm)[1:k-1]
        end
    else
        masks = nothing
    end
    
    return init_positions, masks, motion, positions
end
