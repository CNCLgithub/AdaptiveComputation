Space = Array{Float64}

function render!(cg::CausalGraph)
    graphics = get_graphics(cg)
    render!(cg, graphics)
    return nothing
end

function render!(cg::CausalGraph, graphics::Graphics)
    depth_perm = get_depth_perm(cg, vs)
    spaces =  @>> vs map(v -> render(cg, v, get_prop(cg, v, :object)))
    compose!(spaces, cg, depth_perm)
    for (i, space) in enumerate(spaces)
        set_prop!(cg, vs[i], :space, space)
    end
    return spaces
end


function render!(cg::CausalGraph, v::Int64, d::Dot)
    
    @unpack img_dims = (get_prop(cg, :graphics))
    @unpack area_dims = (get_prop(cg, :gm))
    
    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos[1:2]..., img_dims..., area_dims...)
    scaled_r = d.radius/area_dims[1]*img_dims[1]
    
    @unpack gauss_r_multiple, gauss_amp, gauss_std = (get_prop(cg, :graphics))
    space = draw_gaussian_dot_mask([x,y], scaled_r, img_dims...,
                                   gauss_r_multiple,
                                   gauss_amp, gauss_std)

    flow = has_prop(cg, v, :flow) ? evolve(flow, space) : ExponentialFlow(cg, space)
    set_prop!(cg, v, :flow, flow)

    return flow.memory
end

function render(cg::CausalGraph, v::Int64, e::UniformEnsemble)
    @unpack img_dims = (get_prop(cg, :graphics))
    space = fill(e.pixel_prob, img_dims)
end


function compose!(spaces::Vector{Space}, cg::CausalGraph, depth_perm::Vector{Int64})
    @unpack img_dims = (get_prop(cg, :graphics))
    canvas = zeros(img_dims)
    
    for i in depth_perm
        spaces[i] -= canvas
        canvas += spaces[i]
        clamp!.(spaces[i], 0.0, 1.0)
    end

    return nothing
end

function get_depth_perm(cg::CausalGraph, vs::Vector{Int64})
    @>> vs begin
        map(v -> get_pos(get_prop(cg, v, :object))[3])
        sortperm
    end
end
