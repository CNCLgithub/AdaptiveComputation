abstract type AbstractReceptiveField end
abstract type NullReceptiveFields end # used to indicate absence of RF

export RectangleReceptiveField, get_rectangle_receptive_fields

include("gen.jl")

function crop(rf::AbstractReceptiveField,
              space::Space)
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
              space::Space)
    cs = CartesianIndices(size(space))
    idxs = cs[rf.p1[2]:rf.p2[2], rf.p1[1]:rf.p2[1]]
    space[idxs]
end

"""
    returns height, width
"""
function get_dimensions(rf::RectangleReceptiveField)
    (w, h) = rf.p2 .- rf.p1 .+ (1, 1)
    return (h, w)
end


# """
    # simple, nonhierarchical case
# """
# function get_mask_distributions(objects, gm::GMParams;
                                # flow_masks=nothing)
    # pos = map(o->o.pos, objects)
    # mask_args, trackers_img = get_masks_rvs_args(pos, gm)
    # mask_distributions = @>> mask_args map(first)
    # if !isnothing(flow_masks)
        # flow_masks = update_flow_masks(flow_masks, mask_distributions)
        # mask_distributions = predict(flow_masks)
    # end
    # return (mask_distributions, flow_masks)
# end


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
                                       overlap::Float64)

    p1 = (1+(cidx[2]-1)*(1-overlap)*w, 1+(cidx[1]-1)*(1-overlap)*h) # top left
    p2 = p1 .+ (w-1, h-1) # bottom right
    
    return RectangleReceptiveField(floor.(Int64, p1), floor.(Int64, p2), rf_threshold)
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
                                        overlap::Float64)
    
    # calculating the width and height of each receptive field
    w = floor.(Int64, img_dims[1]/(rf_dims[1] - (rf_dims[1] - 1)*overlap))
    h = floor.(Int64, img_dims[2]/(rf_dims[2] - (rf_dims[2] - 1)*overlap))

    # first index is the row (height) and second index is the col (width)
    rf_cidx = CartesianIndices(reverse(rf_dims))
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
function cropfilter(rf, masks::Vector)
    @>> masks begin
        map(mask -> crop(rf, mask))
        filter(mask -> mean(mask) > rf.threshold)
    end
end

init_rfs_vec(rf_dims) = Vector{RFSElements{Array}}(undef, prod(rf_dims))

