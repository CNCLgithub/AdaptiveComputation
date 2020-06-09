export get_masks

# translates coordinate from euclidean to image space
function translate_area_to_img(x, y, params)
    # inverting y
    y *= -1

    x *= params["img_width"]/params["area_width"]
    x = round(Int, x+params["img_width"]/2)

    y *= params["img_height"]/params["area_height"]
    y = round(Int, y+params["img_height"]/2) 
    
    return x, y
end


# draws a dot and subtracts image so far
function draw_masked_dot(pos, img_so_far, params)
    img_height, img_width = size(img_so_far)
    x, y = translate_area_to_img(pos[1], pos[2], params)
    
    mask = BitArray{2}(undef, img_height, img_width)
    mask .= false

    radius = params["dot_radius"] * img_width / params["area_width"]
    draw_circle!(mask, [x,y], radius, true)
    
    # getting rid of the intersection
    mask[img_so_far] .= false

    return mask
end



"""
    get_masks(positions::Array{Float64})

    returns an array of masks
"""
function get_masks(positions::Array{Float64}, params)
    T, num_dots = size(positions)
    masks = Array{Matrix{Bool}}(undef, T, num_dots)
    
    for t=1:T
        pos = positions[t,:,:]

        # sorting according to depth
        depth_perm = sortperm(pos[:, 3])
        pos = pos[depth_perm, :]

        # initially empty image
        img_so_far = BitArray{2}(undef, params["img_height"], params["img_width"])
        img_so_far .= false

        for i=1:num_dots
            mask = draw_masked_dot(pos[i,:], img_so_far, params)
            masks[t,i] = mask
            img_so_far = mask .| img_so_far
        end
    end

    return masks
end

