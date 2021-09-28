export Graphics

################################################################################
# Graphics
################################################################################

@with_kw struct Graphics <: AbstractGraphics
    img_dims::Tuple{Int64, Int64}
    rf_dims::Tuple{Int64, Int64}
    receptive_fields
    flow_decay_rate::Float64

    # parameters for the drawing the mask random variable arguments
    gauss_r_multiple::Float64 = 4.0 # multiple where to thershold the mask
    gauss_amp::Float64 = 0.8 # gaussian amplitude for the gaussian component of the mask
    gauss_std::Float64 = 1.0 # standard deviation --||--

    bern_existence_prob::Float64 = 0.99
end

"""
    loads from JSON which has to have all the symboled elements
"""
function load(::Type{Graphics}, path::String)
    data = read_json(path)
    img_dims = (data[:img_width], data[:img_height])
    rf_dims = (data[:rf_width], data[:rf_height])
    receptive_fields = get_rectangle_receptive_fields(rf_dims,
                                                      img_dims,
                                                      data[:rf_threshold],
                                                      data[:rf_overlap])
    
    flow_decay_rate = data[:flow_decay_rate]
    gauss_r_multiple, gauss_amp, gauss_std = (data[:gauss_r_multiple], data[:gauss_amp],
                                              data[:gauss_std])
    bern_existence_prob = data[:bern_existence_prob]

    Graphics(img_dims, rf_dims, receptive_fields, flow_decay_rate,
             gauss_r_multiple, gauss_amp, gauss_std, bern_existence_prob)
end

function predict(gr::Graphics, cg::CausalGraph)::Diff
    vs = collect(filter_vertices(cg, :space))
    nvs = length(vs)
    es = RFSElements{BitMatrix}(undef, nvs)
    @inbounds for j in 1:nvs
        es[j] = predict(gr, cg, vs[j], get_prop(cg, vs[j], :object))
    end
    Diff(Dict{ChangeDiff, Any}((:es => :es) => es))
end


################################################################################
# Rendering
################################################################################
"""
`Graphics` does not define any non-self graphical interactions.

Only updates local memory (:flow) and observation space (:space)
"""
function render(gr::Graphics,
                cg::CausalGraph)::Diff
    ch = ChangeDict()
    for v in LightGraphs.vertices(cg)
        @>> get_prop(cg, v, :object) render_elem!(ch, gr, cg, v)
    end
    Diff(Thing[], Int64[], StaticPath[], ch)
end

"""
Catch for undefined graphics
"""
function render_elem!(::ChangeDict,
                      ::Graphics,
                      ::CausalGraph,
                      ::Int64,
                      ::Thing)
    return nothing
end

"""
Rendering Dots
"""
function render_elem!(ch::ChangeDict,
                      gr::Graphics,
                      cg::CausalGraph,
                      v::Int64,
                      d::Dot)

    @unpack img_dims, gauss_r_multiple, gauss_amp, gauss_std = gr
    @unpack area_width, area_height = (get_prop(cg, :gm))

    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos[1:2]...,
                                 img_dims..., area_width, area_height)
    scaled_r = d.radius/area_width*img_dims[1]

    space = draw_gaussian_dot_mask([x,y], scaled_r, img_dims...,
                                   gauss_r_multiple,
                                   gauss_amp, gauss_std)

    if has_prop(cg, v, :flow)
        flow = evolve(get_prop(cg, v, :flow), space)
    else
        @unpack flow_decay_rate = gr
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
function render_elem!(ch::ChangeDict,
                      gr::Graphics,
                      cg::CausalGraph,
                      v::Int64,
                      e::UniformEnsemble)
    @unpack img_dims = gr
    ch[v => :space] = Fill(e.pixel_prob, reverse(img_dims))
end

################################################################################
# Prediction
################################################################################
function predict(gr::Graphics, cg::CausalGraph, v::Int64, e::Dot)
    ep = gr.bern_existence_prob
    space = get_prop(cg, v, :space)
    BernoulliElement{BitMatrix}(ep, mask, (space,))
end

function predict(gr::Graphics, cg::CausalGraph, v::Int64, e::UniformEnsemble)
    space = get_prop(cg, v, :space)
    PoissonElement{BitMatrix}(e.rate, mask, (space,))
end


################################################################################
# Helpers
################################################################################

function render_from_cgs(gr::Graphics,
                         gm::GMParams,
                         cgs::Vector{CausalGraph})
    k = length(cgs)
    # time x thing
    # first time step is initialization (not inferred)
    bit_masks= Vector{Vector{BitMatrix}}(undef, k-1)

    # initialize graphics
    g = first(cgs)
    set_prop!(g, :gm, gm)
    gr_diff = render(gr, g)
    @inbounds for t = 2:k
        g = cgs[t]
        set_prop!(g, :gm, gm)
        # carry over graphics from last step
        patch!(g, gr_diff)
        # render graphics from current step
        gr_diff = render(gr, g)
        patch!(g, gr_diff)
        @>> g predict(gr) patch!(g)
        # create masks
        bit_masks[t - 1] = rfs(get_prop(g, :es))
    end
    bit_masks
end
