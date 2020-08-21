export get_masks_from_mask_rcnn


function get_masks_from_mask_rcnn(positions, gm)
    # rendering images in memory (array=true)
    imgs = render(positions, gm,
                  stimuli=true,
                  array=true)

    println("getting those masks from Mask RCNN...")

    masks = Vector{Vector{BitArray{2}}}(undef, k)
    for t=1:k-1
        print("timestep: $t / $(k) \r")
        masks_t = []
        chan_img = channelview(RGB.(imgs[t]))
        masks_bool = mask_rcnn.get_masks(chan_img)
        for i=1:size(masks_bool,1)
            mask_bool = transpose(masks_bool[i,:,:])
            mask_bool = imresize(mask_bool, gm.img_height, gm.img_width)
            mask_bool = round.(Int, mask_bool)
            push!(masks_t, BitArray(mask_bool))
        end
        masks[t] = masks_t
    end

    println("Mask RCNN done!")

    return masks
end
