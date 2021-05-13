abstract type AbstractGraphics end

load(::Type{AbstractGraphics}) = error("not implemented")

get_observations(::AbstractGraphics) = error("not implemented")

@with_kw struct Graphics <: AbstractGraphics
    img_dims::Tuple{Int64, Int64}
    receptive_fields
    flow
end

function load(::Type{Graphics}, path::String)
    data = read_json(path)
    @unpack img_dims, rf_dims, rf_threshold, overlap, decay_rate = data
    receptive_fields = get_rectangle_receptive_field(rf_dims,
                                                     img_dims,
                                                     rf_threshold, overlap)
    flow = ExponentialFlow(decay_rate, zeros(img_dims))

    Graphics(img_dims, receptive_fields, flow)
end


include("space.jl")

function predict(graphics::Graphics, e::Dot, space::Space)
    BernoulliElement{Array}(graphics.bern_existence_prob, mask, space)
end

function predict(graphics::Graphics, e::UniformEnsemble, space::Space)
    PoissonElement{Array}(e.rate, mask, (space,))
end

function graphics_update!(cg::CausalGraph)
    graphics = get_graphics(cg)
    graphics_update!(cg, graphics)
end

function graphics_update!(cg::CausalGraph, graphics::Graphics)
    vs = get_prop(cg, :graphics_vs)
    spaces = render!(cg, vs) # project to graphical space

    spaces_rf = @>> graphics.receptive_fields begin
        map(rf -> get_mds_rf(rf, spaces))
    end
    
    rfs_vec = init_rfs_vec(graphics.rf_dims)
    for i in LinearIndices(rf_dims)
        rfes = RFSElements{Array}(undef, length(spaces_rf[i]))
        for (j, space_rf) in enumerate(spaces_rf[i])
            rfes[j] = predict(graphics, get_prop(cg, vs[j], :object), space_rf)
            #set_prop!(cg, vs[j], :rfe, rfes[j])
        end
        rfs_vec[i] = rfes
    end
    
    return rfs_vec
end

function graphics_init!(cg::CausalGraph)
    graphics = get_graphics(cg)
    graphics_init!(cg, graphics)
    return nothing
end

function graphics_init!(cg::CausalGraph, graphics::Graphics)
    vs = @> cg begin
        filter_vertices((g, v) -> get_prop(g, v, :object) isa
                        Union{Dot, UniformEnsemble})
    end
    set_prop!(cg, :graphics_vs, vs)
    predict!(cg, graphics)
    return nothing
end



include("utils.jl")
include("shapes.jl")
include("masks.jl")
include("receptive_fields/receptive_fields.jl")
include("flow.jl")


