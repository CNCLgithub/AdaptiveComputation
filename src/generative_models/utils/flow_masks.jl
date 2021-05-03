struct FlowMasks
    masks::Matrix{Matrix{Float64}}
    decay_function::Function
end

function FlowMasks(n_trackers::Int64, gm::AbstractGMParams)
    masks = fill(zeros(gm.img_height, gm.img_width), gm.fmasks_n, n_trackers)
    FlowMasks(masks, gm.fmasks_decay_function)
end

function default_decay_function(mask, decay_rate)
    mask .* exp(decay_rate)
end

function predict(flow_masks::FlowMasks)
    n_masks, n_trackers = size(flow_masks.masks)
    mask_distributions = Vector{Matrix{Float64}}(undef, n_trackers)

    for i=1:n_trackers
        md = flow_masks.masks[n_masks,i]
        for j=1:n_masks-1
            fmask = flow_masks.masks[n_masks-j,i]
            fmask = subtract_images(fmask, md)
            md = add_images(fmask, md)
        end
        mask_distributions[i] = md
    end
    return mask_distributions
end

function update_flow_masks(flow_masks::FlowMasks, new_masks::Vector{Matrix{Float64}})
    masks = copy(flow_masks.masks)
    masks = circshift(masks, (-1, 0))
    masks = flow_masks.decay_function.(masks)
    masks[end,:] = new_masks
    
    clamp!.(masks, 1e-5, 1.0 - 1e-5)

    return FlowMasks(masks, flow_masks.decay_function)
end
