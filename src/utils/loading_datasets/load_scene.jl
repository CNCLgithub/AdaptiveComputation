export load_scene

function try_read(scene, element::String)
    datum = nothing
    try
        datum = scene[element]
    catch
    end
    return datum
end

"""
loads gt causal graphs and motion
"""
function load_scene(scene, dataset_path, gm;
                    generate_masks=true,
                    from_mask_rcnn=false)
    
	file = jldopen(dataset_path, "r")
    scene = read(file, "$scene")
    #dm = scene["dm"]
    gt_causal_graphs = scene["gt_causal_graphs"]
    
    # new entry in scene data, perhaps try block
    # would be good
    gm = try_read(scene, "gm")
    dm = try_read(scene, "dm")
    aux_data = try_read(scene, "aux_data")
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
        flow_masks = FlowMasks(Int64(gm.n_trackers + gm.distractor_rate), gm)

        for t=1:length(masks)
            masks_float = convert(Vector{Matrix{Float64}}, masks[t])
            flow_masks = update_flow_masks(flow_masks, masks_float)
            mask_distributions = predict(flow_masks)
            masks[t] = @>> mask_distributions map(mask)
        end
    end
    
    println("scene data loaded")
    
    scene_data = Dict([:gt_causal_graphs => gt_causal_graphs,
                       :dm => dm,
                       :gm => gm,
                       :masks => masks,
                       :aux_data => aux_data])
    
    return scene_data
end
