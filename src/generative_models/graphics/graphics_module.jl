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

function predict(graphics::Graphics, cg::CausalGraph)::Diff

    # # cut each mass matrix into each receptive fiel
    # spaces_rf = @>> graphics.receptive_fields begin
    #     map(rf -> cropfilter(rf, spaces))
    # end
    #
    vs = collect(filter_vertices(cg, :space))
    nvs = length(vs)

    # construct receptive fields
    rfs_vec = init_rfs_vec(graphics.rf_dims)
    @inbounds for i in LinearIndices(graphics.rf_dims)
        rfes = RFSElements{BitMatrix}(undef, nvs)
        for j in 1:nvs
            rfes[j] = predict(cg, vs[j], get_prop(cg, vs[j], :object))
        end
        rfs_vec[i] = rfes
    end

    Diff(Dict{ChangeDiff, Any}((:rfs_vec => :rfv_vec) => rfs_vec))
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
function predict(cg::CausalGraph, v::Int64, e::Dot)
    ep = get_graphics(cg).bern_existence_prob
    space = get_prop(cg, v, :space)
    BernoulliElement{BitMatrix}(ep, mask, (space,))
end

function predict(cg::CausalGraph, v::Int64, e::UniformEnsemble)
    space = get_prop(cg, v, :space)
    PoissonElement{BitMatrix}(e.rate, mask, (space,))
end


################################################################################
# Helpers
################################################################################

# Returns a Vector{BitMatrix} with dots drawn according to the causal graph
function get_bit_masks(cg::CausalGraph,
                       graphics::Graphics,
                       gm::GMParams)


    @unpack img_dims, gauss_amp, gauss_std, gauss_r_multiple = graphics
    # @unpack dot_radius, area_width, area_height = (get_prop(cg, :gm))
    @unpack dot_radius, area_width, area_height = gm

    positions = @>> get_objects(cg, Dot) map(x -> x.pos)

    # sorting according to depth
    depth_perm = sortperm(map(x->x[3], positions))
    positions = positions[depth_perm]

    # initially empty image
    img_so_far = BitArray{2}(zeros(reverse(graphics.img_dims)))

    n_objects = size(positions,1)
    masks = Vector{SparseMatrixCSC}(undef, n_objects)

    scaled_r = (dot_radius / area_width) * img_dims[1]

    for i=1:n_objects
        x, y = translate_area_to_img(positions[i][1:2]...,
                                     img_dims..., area_width, area_height)
        masks[i] = draw_gaussian_dot_mask([x,y], scaled_r, img_dims...,
                                          gauss_r_multiple,
                                          gauss_amp, gauss_std)
    end

    masks = masks[invperm(depth_perm)]
    masks
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
    # time x receptive_field x object
    bit_masks_rf = Vector{Vector{Vector{BitMatrix}}}(undef, k)

    vs = @> first(cgs) begin
        get_objects(Dot)
    end
    n_objects = length(vs)

    @unpack img_dims, flow_decay_rate, gauss_amp = graphics
    flows = @>> vs begin
        map(v -> ExponentialFlow(decay_rate = flow_decay_rate,
                                 memory = spzeros(Float64, reverse(img_dims)...)))
        collect(ExponentialFlow)
    end

    for t=1:k
        # first create the amodal mask for each object
        bit_masks = Vector{BitMatrix}(undef, n_objects)
        for (i, m) in enumerate(get_bit_masks(cgs[t], graphics, gm))
            flows[i] = evolve(flows[i], m) # evolve the flow
            bit_masks[i] = mask(flows[i].memory) # mask is the composed flow thing
        end
        # then parse each mask across receptive fields
        bit_masks_rf[t] = @>> graphics.receptive_fields begin
            map(rf -> cropfilter(rf, bit_masks))
        end
        @debug "# of masks per rf : $(map(length, bit_masks_rf[t]))"
    end

    bit_masks_rf
end
