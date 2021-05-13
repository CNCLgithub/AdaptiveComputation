abstract type AbstractReceptiveField end
abstract type NullReceptiveFields end # used to indicate absence of RF

include("receptive_fields_gen.jl")

function crop(rf::AbstractReceptiveField,
              mask_distribution::Union{Matrix{Float64}, BitMatrix})
    println("not implemented")
end

function get_dimensions(::T) where T <: AbstractReceptiveField
    println("not implemented")
end

"""
    parametrized by two points:
    p1 = (xmin, ymin) top left
    p2 = (xmax, ymax) bottom right
"""
struct RectangleReceptiveField <: AbstractReceptiveField
    p1::Tuple{Int64, Int64}
    p2::Tuple{Int64, Int64}
    threshold::Float64
end


RectangleReceptiveFields = Vector{RectangleReceptiveField}

function crop(rf::RectangleReceptiveField,
              mask_distribution::Union{Matrix{Float64}, BitMatrix})
    cs = CartesianIndices(size(mask_distribution))
    idxs = cs[rf.p1[2]:rf.p2[2], rf.p1[1]:rf.p2[1]]
    mask_distribution[idxs]
end

"""
    height, width
"""
function get_dimensions(rf::RectangleReceptiveField)
    (w, h) = rf.p2 .- rf.p1 .+ (1, 1)
    return (h, w)
end

# gets the mask distributions for a given receptive field
function get_mds_rf(rf::AbstractReceptiveField,
                    mds::Vector{Matrix{Float64}},
                    rf_threshold::Float64)
    error("not implemented")
end

function get_mds_rf(rf::RectangleReceptiveField,
                    mds::Vector{Matrix{Float64}})
    # cropping masks to receptive fields and filtering only with mass
    @>> mds map(md -> crop(rf, md)) filter(md -> any(md .> rf.threshold))
end

"""
    simple, nonhierarchical case
"""
function get_mask_distributions(objects, gm::GMParams; flow_masks=nothing)
    pos = map(o->o.pos, objects)
    mask_args, trackers_img = get_masks_rvs_args(pos, gm)
    mask_distributions = @>> mask_args map(first)
    if !isnothing(flow_masks)
        flow_masks = update_flow_masks(flow_masks, mask_distributions)
        mask_distributions = predict(flow_masks)
    end
    return (mask_distributions, flow_masks)
end



###############
# A FEW UTILS for automatic generation of receptive
# fields given generative model params
###############

function bound_point(p, w, h)
    (clamp(p[1], 1, w), clamp(p[2], 1, h))
end

function get_rectangle_receptive_field(cidx::CartesianIndex{2},
                                       w::Int64, h::Int64,
                                       img_dims::Tuple{Int64, Int64},
                                       rf_threshold::Float64,
                                       overlap::Int64)

    p1 = (w*(cidx[2]-1)+1, h*(cidx[1]-1)+1) # top left
    p2 = p1 .+ (w-1, h-1) # bottom right
    
    # add overlap
    p1 = p1 .- (overlap, overlap)
    p2 = p2 .+ (overlap, overlap)
    
    # make sure points are within bounds of the image
    p1 = bound_point(p1, img_dims...)
    p2 = bound_point(p2, img_dims...)

    return RectangleReceptiveField(p1, p2, rf_threshold)
end

"""
    get_rectangle_receptive_fields(rf_dims::Tuple{Int64, Int64},
                                        img_dims::Tuple{Int64, Int64},
                                        overlap::Int64)

    Arguments:
        rf_dims - (n_x, n_y)
        img_dims - (width, height)
        overlap
"""
function get_rectangle_receptive_fields(rf_dims::Tuple{Int64, Int64},
                                        img_dims::Tuple{Int64, Int64},
                                        rf_threshold::Float64,
                                        overlap::Int64)

    w, h = ceil.(Int64, img_dims ./ rf_dims)

    # first index is the row (height) and second index is the col (width)
    rf_cidx = CartesianIndices((n_y, n_x))
    @>> rf_cidx begin
        vec
        map(cidx -> get_rectangle_receptive_field(cidx, w, h, img_dims, rf_threshold, overlap))
    end
end


function Graphics(::Type{RectangleReceptiveField},
                  img_dims::Tuple{Int64, Int64},
                  rf_dims::Tuple{Int64, Int64},
                  rf_threshold::Float64,
                  overlap::Int64)

    receptive_fields = get_rectangle_receptive_fields(rf_dims, img_dims, rf_threshold, overlap)
    
    Graphics(img_dims, receptive_fields, flow_decay_rate)
end


"""
 crop masks to receptive fields and then
 filter so each mask is non zero
"""
function cropfilter(rf, masks)
    @>> masks begin
        map(mask -> crop(rf, mask))
        filter(mask -> !iszero(sum(mask)))
    end
end

init_rfs_vec(rf_dims) = Vector{RFSElements{Array}}(undef, prod(rf_dims))

export RectangleReceptiveField, get_rectangle_receptive_fields
