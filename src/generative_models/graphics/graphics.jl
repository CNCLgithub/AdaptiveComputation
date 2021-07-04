export Graphics

abstract type AbstractGraphics end

load(::Type{AbstractGraphics}) = error("not implemented")

get_observations(::AbstractGraphics) = error("not implemented")

@with_kw struct Graphics <: AbstractGraphics
    img_dims::Tuple{Int64, Int64}
    rf_dims::Tuple{Int64, Int64}
    receptive_fields
    flow_decay_rate::Float64

    # parameters for the drawing the mask random variable arguments
    gauss_r_multiple::Float64 = 2.5 # multiple where to thershold the mask
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

include("space.jl")

function predict(cg::CausalGraph, e::Dot, space::Space)
    ep = get_graphics(cg).bern_existence_prob
    BernoulliElement{BitMatrix}(ep, mask, (space,))
end

function predict(cg::CausalGraph, e::UniformEnsemble, space::Space)
    PoissonElement{BitMatrix}(e.rate, mask, (space,))
end

function graphics_init(cg::CausalGraph)
    g = get_graphics(cg)
    graphics_init(cg, g)
end

function graphics_init(cg::CausalGraph, graphics::Graphics)
    cg = deepcopy(cg)
    vs = @> cg begin
        filter_vertices((g, v) -> get_prop(g, v, :object) isa
                        Union{Dot, UniformEnsemble})
        (@>> collect(Int64))
    end
    set_prop!(cg, :graphics_vs, vs)
    return cg
end

function graphics_update(cg::CausalGraph)
    graphics = get_graphics(cg)
    graphics_update(cg, graphics)
end

function graphics_update(cg::CausalGraph, graphics::Graphics)
    cg = deepcopy(cg)
    vs = get_prop(cg, :graphics_vs)

    # first create the sparse mass matrices for each element
    spaces = render!(cg)

    # cut each mass matrix into each receptive field
    spaces_rf = @>> graphics.receptive_fields begin
        map(rf -> cropfilter(rf, spaces))
    end

    # construct the
    rfs_vec = init_rfs_vec(graphics.rf_dims)
    for i in LinearIndices(graphics.rf_dims)
        rfes = RFSElements{BitMatrix}(undef, length(spaces_rf[i]))
        for (j, space_rf) in enumerate(spaces_rf[i])
            rfes[j] = predict(cg, get_prop(cg, vs[j], :object), space_rf)
        end
        rfs_vec[i] = rfes
    end
    set_prop!(cg, :rfs_vec, rfs_vec)

    return cg
end

include("shapes.jl")
include("masks.jl")
include("receptive_fields/receptive_fields.jl")
include("flow.jl")

