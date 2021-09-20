export NullGraphics, Graphics

################################################################################
# Null Graphics
################################################################################

struct NullGraphics <: AbstractGraphics end

function graphics_init(graphics::NullGraphics, cg::CausalGraph)
    return cg
end

function graphics_update(graphics::NullGraphics, cg::CausalGraph)
    return cg
end


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

function graphics_init(graphics::Graphics, cg::CausalGraph)
    cg = deepcopy(cg)
    vs = @> cg begin
        filter_vertices((g, v) -> get_prop(g, v, :object) isa
                        Union{Dot, UniformEnsemble})
        (@>> collect(Int64))
    end
    set_prop!(cg, :graphics_vs, vs)
    return cg
end

function predict(graphics::Graphics, cg::CausalGraph)::Diff

    # # cut each mass matrix into each receptive fiel
    # spaces_rf = @>> graphics.receptive_fields begin
    #     map(rf -> cropfilter(rf, spaces))
    # end
    #
    vs = filter_vertices(cg, :space)
    nvs = length(rendered_vs)

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
# Prediction
################################################################################

include("space.jl")

function predict(cg::CausalGraph, v::Int64, e::Dot)
    ep = get_graphics(cg).bern_existence_prob
    space = get_prop(cg, v, :space)
    BernoulliElement{BitMatrix}(ep, mask, (space,))
end

function predict(cg::CausalGraph, v::Int64, e::UniformEnsemble)
    space = get_prop(cg, v, :space)
    PoissonElement{BitMatrix}(e.rate, mask, (space,))
end


# include("shapes.jl")
include("flow.jl")
include("masks.jl")
include("receptive_fields/receptive_fields.jl")
