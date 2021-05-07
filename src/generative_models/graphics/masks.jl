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
function two_dimensional_gaussian(x::I, y::I, x_0::T, y_0::T, A::T,
                                  sigma_x::T, sigma_y::T) where
    {I<:Int64,T<:Float64}
    A * exp(-( (x-x_0)^2/(2*sigma_x^2) + (y-y_0)^2/(2*sigma_y^2)))
end


"""
drawing a gaussian dot with two components:
1) just a dot at the center with probability 1 and 0 elsewhere
2) spread out gaussian modelling where the dot is likely to be in some sense
    and giving some gradient if the tracker is completely off
"""
function draw_gaussian_dot_mask(center::Vector{Float64},
                                r::Real, h::Int, w::Int,
                                gauss_r_multiple::Float64,
                                gauss_amp::Float64, gauss_std::Float64)
    scaled_sd = r * gauss_std
    threshold = r * gauss_r_multiple
    # mask = zeros(h, w)
    mask = fill(1e-10, h, w)
    for i=1:w
        for j=1:h
            (sqrt((i - center[1])^2 + (j - center[2])^2) > threshold) && continue
            mask[j,i] += two_dimensional_gaussian(i, j, center[1], center[2],
                                                  gauss_amp, scaled_sd, scaled_sd)
        end
    end
    mask
end



"""
    get_masks(positions::Array{Float64})

    returns an array of masks
    args:
    r - radius of the dot in image size
    h - image height
    w - image width
    ah - area height
    aw - area width
    ;
    background - true if you want background masks
"""
function get_masks(positions::Vector{Array{Float64}}, r, h, w, ah, aw;
                   background=false)
    k = length(positions)
    masks = Vector{Vector{BitArray{2}}}(undef, k)
    
    for t=1:k
        print("get_masks timestep: $t \r")
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

        masks_t = masks_t[invperm(depth_perm)]
    
        if background
            # pushing background to the end
            bg = BitArray{2}(undef, h, w)
            bg .= true
            bg -= img_so_far
            prepend!(masks_t, [bg])
        end

        masks[t] = masks_t
    end

    return masks
end


function get_positions(gm::AbstractGMParams, cg::CausalGraph)
    positions = @>> vertices(cg) begin
        map(v -> get_prop(cg, v, :object))
        filter(x -> x isa Dot)
        map(x -> x.pos)
    end
end

"""
    get_masks(cgs::Vector{CausalGraph})

    returns an array of masks

    args:
    cgs::Vector{CausalGraph} - causal graphs describing the scene
    gm - generative model parameters
    gm has to include:
    area_width, area_height, img_width, img_height
    ;
    background - true if you want background masks
"""
function get_masks(cgs::Vector{CausalGraph}, gm;
                   background=false)
    k = length(cgs)
    masks = Vector{Vector{BitArray{2}}}(undef, k)
    
    for t=1:k
        print("get_masks timestep: $t / $k \r")
        positions = get_positions(gm, cgs[t])

        # sorting according to depth
        depth_perm = sortperm(map(x->x[3], positions))
        positions = positions[depth_perm]

        # initially empty image
        img_so_far = BitArray{2}(undef, gm.img_height, gm.img_width)
        img_so_far .= false
        
        masks_t = []
        n_objects = size(positions,1)
        for i=1:n_objects
            mask = draw_dot_mask(positions[i], gm.dot_radius,
                                 gm.img_height, gm.img_width,
                                 gm.area_height, gm.area_width)
            mask[img_so_far] .= false
            push!(masks_t, mask)
            img_so_far .|= mask
        end

        masks_t = masks_t[invperm(depth_perm)]
    
        if background
            # pushing background to the end
            bg = BitArray{2}(undef, h, w)
            bg .= true
            bg -= img_so_far
            prepend!(masks_t, [bg])
        end

        masks[t] = masks_t
    end

    return masks
end

