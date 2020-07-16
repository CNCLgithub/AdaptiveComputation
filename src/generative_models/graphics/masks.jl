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
drawing a gaussian dot with two components:
1) just a dot at the center with probability 1 and 0 elsewhere
2) spread out gaussian modelling where the dot is likely to be in some sense
    and giving some gradient if the tracker is completely off
"""
function draw_gaussian_dot_mask(center::Vector{Float64},
                                r::Real, h::Int, w::Int,
                                gauss_amp::Float64, gauss_std::Float64)
 
    mask = zeros(h, w)
    for i=1:h
        for j=1:w
            mask[j,i] += norm([i,j] - center) < r
            mask[j,i] += two_dimensional_gaussian(i, j, center[1], center[2],
                                                  gauss_amp, gauss_std, gauss_std)
        end
    end
    mask = min.(mask, 0.75) # 1.0 - 1e-5)
end



"""
    get_masks(positions::Array{Float64})

    returns an array of masks
"""
function get_masks(positions::Vector{Array{Float64}}, r, h, w, ah, aw)
    k = length(positions)
    masks = Vector{Vector{BitArray{2}}}(undef, k)
    
    for t=1:k
        pos = positions[t]

        # sorting according to depth
        depth_perm = sortperm(pos[:, 3])
        pos = pos[depth_perm, :]

        # initially empty image
        img_so_far = BitArray{2}(undef, h, w)
        img_so_far .= false
        
        masks_t = []
        n_dots = size(pos,1)
        for i=1:n_dots
            mask = draw_dot_mask(pos[i,:], r, h, w, ah, aw)
            mask[img_so_far] .= false
            push!(masks_t, mask)
            img_so_far .|= mask
        end
        masks[t] = masks_t[invperm(depth_perm)]
    end

    return masks
end

