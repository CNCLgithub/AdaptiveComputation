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
    #flow = ExponentialFlow(data[:flow_decay_rate], zeros(img_dims))
    
    flow_decay_rate = data[:flow_decay_rate]
    gauss_r_multiple, gauss_amp, gauss_std = (data[:gauss_r_multiple], data[:gauss_amp],
                                              data[:gauss_std])
    bern_existence_prob = data[:bern_existence_prob]

    Graphics(img_dims, rf_dims, receptive_fields, flow_decay_rate,
             gauss_r_multiple, gauss_amp, gauss_std, bern_existence_prob)
end

include("space.jl")

function get_decayed_existence_prob(cg::CausalGraph, e::Dot)
    graphics = get_graphics(cg)
    @unpack area_width, area_height = get_gm(cg)

    # going from area dims to img dims
    x, y = translate_area_to_img(get_pos(e)[1:2]..., graphics.img_dims...,
                                 area_width, area_height)
    
    # @show x, y
    # @>> graphics.receptive_fields foreach(display)

    # we find the receptive_field
    rf = @>> graphics.receptive_fields begin
        filter(rf -> (rf.p1[1] <= x && rf.p1[2] <= y &&
                      x <= rf.p2[1] + 1 && y <= rf.p2[2] + 1))
        first
    end
    
    # simplified way to find distance to receptive_field walls
    a = x - rf.p1[1]
    b = y - rf.p1[2]
    c = rf.p2[1] - x
    d = rf.p2[2] - y

    decayed = @>> [a,b,c,d] begin
        map(abs)
        minimum
        x -> -x/1000
        exp
    end
    #@show decayed

    # hah a mixture yet again
    return decayed * 0.0 + (1.0 - decayed) * graphics.bern_existence_prob
end

function predict(cg::CausalGraph, e::Dot, space::Space)
    #ep = get_decayed_existence_prob(cg, e)
    ep = get_graphics(cg).bern_existence_prob
    #@show ep
    BernoulliElement{Array}(ep, mask, (space,))
end

function predict(cg::CausalGraph, e::UniformEnsemble, space::Space)
    PoissonElement{Array}(e.rate, mask, (space,))
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

    spaces = render!(cg) # project to graphical space
    spaces_rf = @>> graphics.receptive_fields begin
        map(rf -> get_mds_rf(rf, spaces))
    end
    
    rfs_vec = init_rfs_vec(graphics.rf_dims)
    for i in LinearIndices(graphics.rf_dims)
        rfes = RFSElements{Array}(undef, length(spaces_rf[i]))
        for (j, space_rf) in enumerate(spaces_rf[i])
            rfes[j] = predict(cg, get_prop(cg, vs[j], :object), space_rf)
        end
        rfs_vec[i] = rfes
    end
    set_prop!(cg, :rfs_vec, rfs_vec)

    return cg
end

include("utils.jl")
include("shapes.jl")
include("masks.jl")
include("receptive_fields/receptive_fields.jl")
include("flow.jl")

