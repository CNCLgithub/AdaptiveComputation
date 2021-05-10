abstract type AbstractReceptiveField end

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
end

function crop(rf::RectangleReceptiveField,
              mask_distribution::Union{Matrix{Float64}, BitMatrix})
    cs = CartesianIndices(size(mask_distribution))
    idxs = cs[rf.p1[2]:rf.p2[2], rf.p1[1]:rf.p2[1]]
    # display(mask_distribution[idxs])
    # display(mask_distribution)
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
                    rf_prob_threshold::Float64)
    error("not implemented")
end

function get_mds_rf(rf::RectangleReceptiveField,
                    mds::Vector{Matrix{Float64}},
                    rf_prob_threshold::Float64)
    # cropping masks to receptive fields and filtering only with mass
    @>> mds map(md -> crop(rf, md)) filter(md -> any(md .> rf_prob_threshold))
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


"""
    gets
"""
function get_pmbrfs(rf::AbstractReceptiveField,
                    mds::Vector{Matrix{Float64}}, # maybe define VecMat
                    gm::AbstractGMParams)
    existence_prob = 0.99 # TODO remove constant?

    n = length(mds) + 1 # |mbrfs| + 1 for PPP

    # this is not completely correct, but maybe a fine approximation
    rf_n_pixels = prod(get_dimensions(rf))
    rf_proportion_of_img = rf_n_pixels/gm.img_width*gm.img_height
    rf_distractor_rate = gm.distractor_rate / rf_proportion_of_img

    radius_scaled = gm.dot_radius/gm.area_width*gm.img_width
    clutter_pixel_prob = (rf_distractor_rate*pi*(radius_scaled)^2) / (rf_n_pixels)
    clutter_mask = fill(clutter_pixel_prob, get_dimensions(rf)...)

    pmbrfs = RFSElements{Array}(undef, n)
    pmbrfs[1] = PoissonElement{Array}(rf_distractor_rate, mask, (clutter_mask,))
    for i=2:n
        pmbrfs[i] = BernoulliElement{Array}(existence_prob, mask, (mds[i-1],))
    end
    
    return pmbrfs
end
                    

# gets the vector of random finite sets for each receptive field
function get_rfs_vec(rec_fields::Vector{T},
                     objects::Vector{Object},
                     rf_prob_threshold::Float64,
                     gm::AbstractGMParams;
                     flow_masks=nothing) where T <: AbstractReceptiveField
    mds, flow_masks = get_mask_distributions(objects, gm, flow_masks=flow_masks)
    mds_rf = map(rf -> get_mds_rf(rf, mds, rf_prob_threshold), rec_fields)
    rfs_vec = map(get_pmbrfs, rec_fields, mds_rf, fill(gm, length(rec_fields)))
    return rfs_vec, flow_masks
end


###############
# A FEW UTILS for automatic generation of receptive
# fields given generative model params
###############

function bound_point(p, w, h)
    (clamp(p[1], 1, w), clamp(p[2], 1, h))
end

function get_rectangle_receptive_field(cidx::CartesianIndex{2},
                                       w::Int64, h::Int64, gm;
                                       overlap = 0)

    p1 = (w*(cidx[2]-1)+1, h*(cidx[1]-1)+1) # top left
    p2 = p1 .+ (w-1, h-1) # bottom right
    
    # add overlap
    p1 = p1 .- (overlap, overlap)
    p2 = p2 .+ (overlap, overlap)
    
    # make sure points are within bounds of the image
    p1 = bound_point(p1, gm.img_width, gm.img_height)
    p2 = bound_point(p2, gm.img_width, gm.img_height)

    return RectangleReceptiveField(p1, p2)
end

"""
    get_rectangle_receptive_fields(n_x::Int64, n_y::Int64, gm;
                                    overlap = 0)

    Arguments:
        n_x - number of receptive fields in the x dimension
        n_y - number of receptive fields in the y dimension
        gm - generative model parameters
"""
function get_rectangle_receptive_fields(n_x::Int64, n_y::Int64, gm;
                                        overlap = 0)
    w = ceil(Int64, gm.img_width/n_x) # width of each receptive field
    h = ceil(Int64, gm.img_height/n_y) # height of --||--

    # first index is the row (height) and second index is the col (width)
    rf_cidx = CartesianIndices((n_y, n_x))
    @>> rf_cidx begin
        vec
        map(cidx -> get_rectangle_receptive_field(cidx, w, h, gm; overlap=overlap))
    end
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

export RectangleReceptiveField, get_rectangle_receptive_fields
