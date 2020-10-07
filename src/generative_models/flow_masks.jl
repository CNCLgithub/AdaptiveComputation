struct FlowMasks
    masks::Array{Matrix{Float64}}
    decay_function::Function
end

function add_flow_masks(flow_masks::FlowMasks, masks)
    img_height, img_width = size(first(first(masks)))
    n_trackers, n_fmasks = size(flow_masks.masks)
    new_masks = Vector{Tuple{Array{Float64}}}(undef, length(masks))
    new_fmasks = Array{Matrix{Float64}}(undef, n_trackers, n_fmasks)

    # going through trackers
    for i=1:n_trackers
        new_fmasks[i,1] = masks[i][1]
        new_fmasks[i,2:end] = flow_masks.masks[i,1:end-1]

        mask = new_fmasks[i,1]
        # going through time
        for j=1:n_fmasks-1
            fmask = flow_masks.decay_function(new_fmasks[i,j], j)
            fmask = subtract_images(fmask, mask)
            mask = add_images(fmask, mask)
        end
        # TODO remove
        #mask = zeros(img_height, img_width)
        new_masks[i] = (mask,)
    end
    
    return FlowMasks(new_fmasks, flow_masks.decay_function), new_masks
end
