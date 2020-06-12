export get_masks,
        draw_masked_dot,
        draw_gaussian_dot

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

# 2d gaussian function
function two_dimensional_gaussian(x, y, x_0, y_0, A, sigma_x, sigma_y)
    return A * exp(-( (x-x_0)^2/(2*sigma_x^2) + (y-y_0)^2/(2*sigma_y^2)))
end

# drawing a gaussian dot
function draw_gaussian_dot(center, graphics_params)
    radius = graphics_params["dot_radius"]

    # standard deviation based on the volume of the Gaussian
    spread_1 = 1.0 # parameter for how spread out the mask is
    spread_2 = 5.0
    A = 0.4999999999
    std_1 = sqrt(spread_1 * radius)
    std_2 = sqrt(spread_2 * radius)

    #A = 0.5/spread^2
    
    img = zeros(graphics_params["img_height"], graphics_params["img_width"])
    for i=1:size(img,1)
        for j=1:size(img,2)
            img[j,i] = two_dimensional_gaussian(i, j, center[1], center[2], A, std_1, std_1) 
            img[j,i] += two_dimensional_gaussian(i, j, center[1], center[2], A, std_2, std_2) 
        end
    end
    
    return img
end



"""
    get_masks(positions::Array{Float64})

    returns an array of masks
"""
function get_masks(positions::Array{Float64}, params)
    T, num_dots = size(positions)
    masks = Array{BitArray{2}}(undef, T, num_dots)
    
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

