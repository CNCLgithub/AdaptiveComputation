export get_masks,
        draw_dot_mask,
        draw_gaussian_dot_mask,
        translate_area_to_img,
        generate_masks

# translates coordinate from euclidean to image space
function translate_area_to_img(x, y, img_width, img_height,
                               area_width, area_height)

    x *= img_width/area_width
    x += img_width/2

    # inverting y
    y *= -1 * img_height/area_height
    y += img_height/2
    
    return x, y
end


# draws a dot
function draw_dot_mask(pos, r, w, h, aw, ah)
    x, y = translate_area_to_img(pos[1], pos[2], w, h, aw, ah)
    
    mask = BitMatrix(zeros(h, w))
    
    radius = ceil(r * w / aw)

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
                                r::Real, w::Int, h::Int,
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


function get_bit_masks(cg::CausalGraph,
                       graphics::Graphics,
                       gm::AbstractGMParams)

    positions = @>> get_objects(cg, Dot) map(x -> x.pos)

    # sorting according to depth
    depth_perm = sortperm(map(x->x[3], positions))
    positions = positions[depth_perm]

    # initially empty image
    img_so_far = BitArray{2}(zeros(reverse(graphics.img_dims)))

    n_objects = size(positions,1)
    masks = Vector{BitMatrix}(undef, n_objects)

    for i=1:n_objects
        mask = draw_dot_mask(positions[i], gm.dot_radius,
                             graphics.img_dims...,
                             gm.area_width, gm.area_height)
        mask[img_so_far] .= false
        masks[i] = mask
        img_so_far .|= mask
    end

    masks = masks[invperm(depth_perm)]
    masks
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
function get_bit_masks(cgs::Vector{CausalGraph},
                       graphics::AbstractGraphics,
                       gm::AbstractGMParams;
                       background=false)

    k = length(cgs)
    masks = Vector{Vector{BitMatrix}}(undef, k)
    
    for t=1:k
        @debug "get_masks timestep: $t / $k \r"
        masks[t] = get_bit_masks(cgs[t], graphics, gm)
    end

    return masks
end


function generate_masks(cgs::Vector{CausalGraph},
                        graphics::Graphics,
                        gm::AbstractGMParams)
    k = length(cgs)
    bit_masks = get_bit_masks(cgs, graphics, gm)
    # time x receptive_field x object
    bit_masks_rf = Vector{Vector{Vector{BitMatrix}}}(undef, k)

    vs = @> first(cgs) begin
        filter_vertices((g, v) -> get_prop(g, v, :object) isa Dot)
    end

    @unpack img_dims, flow_decay_rate, gauss_amp = graphics
    decay_rate = graphics.flow_decay_rate
    flows = @>> vs begin
        map(v -> ExponentialFlow(decay_rate,
                                 zeros(reverse(img_dims)),
                                 gauss_amp))
        collect(ExponentialFlow)
    end

    for t=1:k
        # first create the amodal mask for each object
        for (i, m) in enumerate(bit_masks[t])
            flows[i] = evolve(flows[i], convert(Matrix{Float64}, m)) # evolve the flow
            bit_masks[t][i] = mask(flows[i].memory) # mask is the composed flow thing
        end
        # then parse each mask across receptive fields
        bit_masks_rf[t] = @>> graphics.receptive_fields begin
            map(rf -> cropfilter(rf, bit_masks[t]))
        end
        @debug "# of masks per rf : $(map(length, bit_masks_rf[t]))"
    end

    bit_masks_rf
end
