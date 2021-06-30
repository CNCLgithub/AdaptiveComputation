"""
    Describes how objets get transformed to observation Space
"""

Space{T,N} = AbstractArray{T,N}

function render!(cg::CausalGraph)
    graphics = get_graphics(cg)
    spaces = render!(cg, graphics)
end

function render!(cg::CausalGraph, graphics::Graphics)
    vs = get_prop(cg, :graphics_vs)
    depth_perm = get_depth_perm(cg, vs)
    spaces =  @>> vs begin
        map(v -> render!(cg, v, get_prop(cg, v, :object)))
        collect(Space)
    end
    # compose!(spaces, cg, depth_perm)
    for (i, space) in enumerate(spaces)
        set_prop!(cg, vs[i], :space, space)
    end

    return spaces
end

function render!(cg::CausalGraph, v::Int64, d::Dot)

    @unpack img_dims = (get_prop(cg, :graphics))
    @unpack area_width, area_height = (get_prop(cg, :gm))
    
    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos[1:2]...,
                                 img_dims..., area_width, area_height)
    scaled_r = d.radius/area_width*img_dims[1]
    
    @unpack gauss_r_multiple, gauss_amp, gauss_std = (get_prop(cg, :graphics))
    space = draw_gaussian_dot_mask([x,y], scaled_r, img_dims...,
                                   gauss_r_multiple,
                                   gauss_amp, gauss_std)

    if has_prop(cg, v, :flow)
        flow = evolve(get_prop(cg, v, :flow), space)
    else
        @unpack flow_decay_rate = (get_prop(cg, :graphics))
        flow = ExponentialFlow(flow_decay_rate, space, gauss_amp)
    end
    set_prop!(cg, v, :flow, flow)
    
    return flow.memory
end

function render!(cg::CausalGraph, v::Int64, e::UniformEnsemble)
    @unpack img_dims = (get_prop(cg, :graphics))
    space = fill(e.pixel_prob, reverse(img_dims))
end

# composes the spaces by subtracting occluded parts
function compose!(spaces::Vector{Space}, cg::CausalGraph, depth_perm::Vector{Int64})
    @unpack img_dims = (get_prop(cg, :graphics))
    canvas = zeros(reverse(img_dims)) # reverse to (height, width)
    
    for i in depth_perm
        spaces[i] -= canvas
        canvas += spaces[i]
        clamp!(spaces[i], 1e-10, 1.0 - 1e-10)
    end

    return nothing
end

# returns the permutation according to depth (smallest z values first)
function get_depth_perm(cg::CausalGraph, vs::Vector{Int64})
    @>> vs begin
        map(v -> get_pos(get_prop(cg, v, :object))[3])
        sortperm
    end
end
