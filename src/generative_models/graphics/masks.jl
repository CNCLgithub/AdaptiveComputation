export get_masks,
        draw_dot_mask,
        draw_gaussian_dot_mask,
        translate_area_to_img

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
    radius = round(r * w / aw; digits = 3)
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
    # mask = zeros(h, w) # mask is initially zero to take advantage of sparsity
    x0, y0 = center
    xlim = round.(Int64, [x0 - threshold, x0 + threshold])
    ylim = round.(Int64, [y0 - threshold, y0 + threshold])
    xbounds = clamp.(xlim, 1, w)
    ybounds = clamp.(ylim, 1, h)
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]
    for idx in CartesianIndices((xbounds[1]:xbounds[2],
                                    ybounds[1]:ybounds[2]))
        i,j = Tuple(idx)
        (sqrt((i - x0)^2 + (j - y0)^2) > threshold) && continue
        v = two_dimensional_gaussian(i, j, x0, y0, gauss_amp, scaled_sd, scaled_sd)
        # flip i and j in mask
        push!(Is, j)
        push!(Js, i)
        push!(Vs, v)
    end
    sparse(Is, Js, Vs, h, w)
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

