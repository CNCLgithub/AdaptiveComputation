export load_scene

"""
loads gt causal graphs and motion
"""
function load_scene(scene, dataset_path, gm;
                    generate_masks=true,
                    from_mask_rcnn=false)
    
	file = jldopen(dataset_path, "r")
    scene = read(file, "$scene")
    motion = scene["motion"]
    #gm = scene["gm"]
    gt_causal_graphs = scene["gt_causal_graphs"]
    
    # new entry in scene data, perhaps try block
    # would be good
    
    aux_data = nothing
    try
        aux_data = scene["aux_data"]
    catch
    end
    close(file)

    if generate_masks
        masks = get_masks(gt_causal_graphs[2:end], gm)
        if from_mask_rcnn
            masks[1:k-1] = get_masks_from_mask_rcnn(gt_causal_graphs[2:end],
                                                    gm)[1:k-1]
        end
    else
        masks = nothing
    end
    
    
    if gm.fmasks
        fmasks = Vector{Vector{BitArray{2}}}(undef, length(masks))

        # for each mask, generate a new mask that takes history into account
        for t=1:length(masks)
            print("generating flow masks $t \r")
            fmasks_t = Vector{BitArray{2}}(undef, length(masks[t]))

            # going through individual trackers
            for i=1:length(masks[t])
                new_mask = zeros(gm.img_height, gm.img_width)
                for j=max(1,t-gm.fmasks_n+1):t
                    fmask = masks[t][i]
                    fmask = gm.fmasks_decay_function(fmask, t-j)
                    fmask = subtract_images(fmask, new_mask)
                    new_mask = add_images(fmask, new_mask)
                end

                fmasks_t[i] = mask(new_mask)    
            end
            # sampling a mask
            fmasks[t] = fmasks_t
        end
        masks = fmasks
    end
    
    println("scene data loaded")

    scene_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :motion => motion,
                       #:gm => gm,
                       :masks => masks,
                       :aux_data => aux_data])
end
