export get_masks,
        draw_dot_mask,
        draw_gaussian_dot_mask,
        translate_area_to_img,
        get_bit_masks_rf

# translates coordinate from euclidean to image space
function translate_area_to_img(x::Float64, y::Float64,
                               img_width::Int64, img_height::Int64,
                               area_width::Float64, area_height::Float64)

    x *= img_width/area_width
    x += img_width/2

    # inverting y
    y *= -1 * img_height/area_height
    y += img_height/2
    
    return x, y
end


# Draws a dot mask, i.e. a BitMatrix
function draw_dot_mask(pos::Vector{T},
                       r::T,
                       w::I, h::I,
                       aw::T, ah::T) where {I<:Int64,T<:Float64}

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
function draw_gaussian_dot_mask(center::Vector{T},
                                r::T, w::Int64, h::Int64,
                                gauss_r_multiple::T,
                                gauss_amp::T, gauss_std::T) where {T<:Float64}
    scaled_sd = r * gauss_std
    threshold = r * gauss_r_multiple
    mask = fill(1e-10, h, w)
    # mask = Matrix{Float64}(undef, h, w)
    x0, y0 = center
    xlim = round.(Int64, [x0 - threshold, x0 + threshold])
    ylim = round.(Int64, [y0 - threshold, y0 + threshold])
    xbounds = clamp.(xlim, 1, w)
    ybounds = clamp.(ylim, 1, h)
    for idx in CartesianIndices((xbounds[1]:xbounds[2],
                                    ybounds[1]:ybounds[2]))
        i,j = Tuple(idx)
        (sqrt((i - x0)^2 + (j - y0)^2) > threshold) && continue
        mask[j,i] = two_dimensional_gaussian(i, j, x0, y0, gauss_amp, scaled_sd, scaled_sd)
    end
    mask
end


# Returns a Vector{BitMatrix} with dots drawn according to the causal graph
function get_bit_masks(cg::CausalGraph,
                       graphics::Graphics,
                       gm::GMParams)

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
        masks[i] = mask
    end

    masks = masks[invperm(depth_perm)]
    masks
end


"""
    get_bit_masks(cgs::Vector{CausalGraph},
                       graphics::AbstractGraphics,
                       gm::AbstractGMParams;
                       background=false)

    Returns a Vector{Vector{BitMatrix}} with masks according to
    the Vector{CausalGraph} descrbing the scene

...
# Arguments:
- cgs::Vector{CausalGraph} : causal graphs describing the scene
- graphics : graphical parameters
- gm : generative model parameters
- background : true if you want background masks (i.e.
               1s where there are no objects)
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

"""
    generate_masks(cgs::Vector{CausalGraph},
                        graphics::Graphics,
                        gm::AbstractGMParams)

    Generates masks, adds flow and crops them according to receptive fields.
...
# Arguments:
- cgs::Vector{CausalGraph} : causal graphs describing the scene
- graphics : graphical parameters
- gm : generative model parameters
"""

function get_bit_masks_rf(cgs::Vector{CausalGraph},
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
    flows = @>> vs begin
        map(v -> ExponentialFlow(flow_decay_rate,
                                 zeros(reverse(img_dims)),
                                 # TODO make this a param in graphics
                                 1.0))
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
