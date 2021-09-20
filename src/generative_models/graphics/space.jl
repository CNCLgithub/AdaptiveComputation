"""
    Describes how objets get transformed to observation Space
"""

Space{T,N} = AbstractArray{T,N}

function render(cg::CausalGraph)::Diff
    gr = get_graphics(cg)
    spaces = render(gr, cg)
end

"""
`Graphics` does not define any non-self graphical interactions.

Only updates local memory (:flow) and observation space (:space)
"""
function render(gr::Graphics,
                cg::CausalGraph)::Diff
    ch = Dict{ChangeDiff, Any}()
    for v in LightGraphs.vertices(cg)
        @>> get_prop(cg, v, :object) render_elem!(ch, gr, cg, v)
    end
    Diff(Thing[], Int64[], StaticPath[], ch)
end

"""
Catch for undefined graphics
"""
function render_elem!(::ChangeDiff,
                      ::Graphics,
                      ::CausalGraph,
                      ::Int64,
                      ::Thing)
    return nothing
end

"""
Rendering Dots
"""
function render_elem!(ch::ChangeDiff,
                      gr::Graphics,
                      cg::CausalGraph,
                      v::Int64,
                      d::Dot)

    @unpack img_dims, gauss_r_multiple, gauss_amp, gauss_std = graphics
    @unpack area_width, area_height = (get_prop(cg, :gm))
    
    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos[1:2]...,
                                 img_dims..., area_width, area_height)
    scaled_r = d.radius/area_width*img_dims[1]
    
    space = draw_gaussian_dot_mask([x,y], scaled_r, img_dims...,
                                   gauss_r_multiple,
                                   gauss_amp, gauss_std)

    if has_prop(cg, src, :flow)
        flow = evolve(get_prop(prev_cg, src, :flow), space)
    else
        @unpack flow_decay_rate = graphics
        flow = ExponentialFlow(decay_rate = flow_decay_rate, memory = space)
    end

    # graphics state (internal)
    ch[v => :flow] = flow
    # graphics prediction
    ch[v => :space] = flow.memory

    return nothing
end

"""
Rendering `UniformEnsemble`

No `:flow` needed.
"""
function render_elem!(ch::ChangeDiff,
                      gr::Graphics,
                      cg::CausalGraph,
                      v::Int64,
                      e::UniformEnsemble)
    @unpack img_dims = graphics
    ch[v => :space] = Fill(e.pixel_prob, reverse(img_dims))
end
