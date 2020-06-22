export get_masks,
        draw_dot_mask,
        draw_gaussian_dot_mask,
        translate_area_to_img

# translates coordinate from euclidean to image space
function translate_area_to_img(x, y, img_height, img_width,
                               area_height, area_width;
                               whole_number=true)
    x *= img_width/area_width
    x += img_width/2
    if whole_number
        x = round(Int, x)
    end

    # inverting y
    y *= -1 * img_height/area_height
    y += img_height/2
    if whole_number
        y = round(Int, y)
    end
    
    return x, y
end


# draws a dot
function draw_dot_mask(pos, r, h, w, ah, aw)
    x, y = translate_area_to_img(pos[1], pos[2], h, w, ah, aw)
    
    mask = BitArray{2}(undef, h, w)
    mask .= false

    radius = r * w / aw
    draw_circle!(mask, [x,y], radius, true)
    
    return mask
end


# 2d gaussian function
function two_dimensional_gaussian(x, y, x_0, y_0, A, sigma_x, sigma_y)
    return A * exp(-( (x-x_0)^2/(2*sigma_x^2) + (y-y_0)^2/(2*sigma_y^2)))
end


"""
drawing a gaussian dot
there are two gaussian functions stacked on one another
    parameters for how spread out the mask is
    spread_1: local steep gradient
    spread_2: global gradient
"""
function draw_gaussian_dot_mask(center::Vector{Float64},
                                r::Real, h::Int, w::Int,
                                spread_1::Float64, spread_2::Float64)
    
    # amplitude of first gaussian
    A = 0.4999999999

    std_1 = sqrt(spread_1 * r)
    std_2 = sqrt(spread_2 * r)

    mask = zeros(h, w)
    for i=1:h
        for j=1:w
            mask[j,i] = two_dimensional_gaussian(i, j, center[1], center[2], A, std_1, std_1)
            # mask[j,i] += two_dimensional_gaussian(i, j, center[1], center[2], (1.0-A), std_2, std_2)
            mask[j,i] += two_dimensional_gaussian(i, j, center[1], center[2], A, std_2, std_2)
        end
    end

    return mask
end



"""
    get_masks(positions::Array{Float64})

    returns an array of masks
"""
function get_masks(positions::Array{Float64}, r, h, w, ah, aw)
    k, num_dots = size(positions)
    masks = Vector{Vector{BitArray{2}}}(undef, k)
    
    for t=1:k
        pos = positions[t,:,:]

        # sorting according to depth
        depth_perm = sortperm(pos[:, 3])
        pos = pos[depth_perm, :]

        # initially empty image
        img_so_far = BitArray{2}(undef, h, w)
        img_so_far .= false
        
        masks_t = []
        for i=1:num_dots
            mask = draw_dot_mask(pos[i,:], r, h, w, ah, aw)
            mask[img_so_far] .= false
            push!(masks_t, mask)
            img_so_far .|= mask
        end
        masks[t] = masks_t
    end

    return masks
end

