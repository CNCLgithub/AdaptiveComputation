abstract type AbstractReceptiveField end

function crop(rf::T,
              mask_distribution::Matrix{Float64}) where {T <: AbstractReceptiveField}
    println("not implemented")
end

function get_dimensions(::T) where T <: AbstractReceptiveField
    println("not implemented")
end

"""
    parametrized by two points:
    p1 = (xmin, ymin)
    p2 = (xmax, ymax)
"""
struct RectangleReceptiveField <: AbstractReceptiveField
    p1::Tuple{Int64, Int64}
    p2::Tuple{Int64, Int64}
end

function crop(rf::RectangleReceptiveField,
              mask_distribution::Matrix{Float64})
    idxs = CartesianIndices((rf.p1[1]:rf.p2[1], rf.p1[2]:rf.p2[2]))
    mask_distribution[idxs]
end


function get_dimensions(rf::RectangleReceptiveField)
    (w, h) = rf.p2 .- rf.p1 .+ (1, 1)
end

# gets the mask distributions for a given receptive field
function get_mds_rf(rf::AbstractReceptiveField,
                    mds::Vector{Matrix{Float64}},
                    prob_threshold::Float64)
    error("not implemented")
end

function get_mds_rf(rf::RectangleReceptiveField,
                    mds::Vector{Matrix{Float64}},
                    prob_threshold::Float64)
    # cropping masks to receptive fields and filtering only with mass
    @>> mds map(md -> crop(rf, md)) filter(md -> any(md .> prob_threshold))
    
    # alternative
    # x = @>> mds begin
        # map(md -> crop(rf, md))
        # filter(md -> any(md .> prob_threshold))
    # end
end

"""
    simple, nonhierarchical case
"""
function get_mask_distributions(objects, gm::GMMaskParams; flow_masks=nothing)
    pos = map(o->o.pos, objects)
    mask_args, trackers_img = get_masks_rvs_args(pos, gm)
    if !isnothing(flow_masks)
        flow_masks, mask_args = add_flow_masks(flow_masks, mask_args)
    end
    mask_distributions = map(first, mask_args)
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
                     prob_threshold::Float64,
                     gm::AbstractGMParams;
                     flow_masks=nothing) where T <: AbstractReceptiveField
    mds, flow_masks = get_mask_distributions(objects, gm, flow_masks=flow_masks)
    mds_rf = map(rf -> get_mds_rf(rf, mds, prob_threshold), rec_fields)
    rfs_vec = map(get_pmbrfs, rec_fields, mds_rf, fill(gm, length(rec_fields)))
    return rfs_vec, flow_masks
end


###############
# A FEW UTILS for automatic generation of receptive
# fields given generative model params
###############

function bound(x, a, b)
    min(max(x, a), b)
end

function bound_point(p, w, h)
    (bound(p[1], 1, w), bound(p[2], 1, h))
end

function get_rectangle_receptive_field(xy, n_x, n_y, gm;
                                       overlap = 0)
    w = floor(Int, gm.img_width/n_y)
    h = floor(Int, gm.img_height/n_x)

    p1 = (w*(xy[1]-1)+1, h*(xy[2]-1)+1) .- (overlap, overlap)
    p2 = p1 .+ (w-1, h-1) .+ (overlap, overlap)

    p1 = bound_point(p1, gm.img_width, gm.img_height)
    p2 = bound_point(p2, gm.img_width, gm.img_height)

    return RectangleReceptiveField(p1, p2)
end

"""
    get_rectangle_receptive_fields(n_x, n_y, gm)

    Arguments:
        n_x - number of receptive fields in the x dimension
        n_y - number of receptive fields in the y dimension
        gm - generative model parameters

"""
function get_rectangle_receptive_fields(n_x, n_y, gm;
                                        overlap = 0)
    rf_idx = Iterators.product(1:n_x, 1:n_y)
    receptive_fields = map(xy -> get_rectangle_receptive_field(xy, n_x, n_y, gm; overlap=overlap), rf_idx)
    println(receptive_fields)
    receptive_fields = map(i -> receptive_fields[i], 1:n_x*n_y) # I can't find a way to flatten ://///
end

function crop(rf::RectangleReceptiveField,
              mask_distribution::BitArray{2})
    idxs = CartesianIndices((rf.p1[1]:rf.p2[1], rf.p1[2]:rf.p2[2]))
    mask_distribution[idxs]
end

export RectangleReceptiveField, get_rectangle_receptive_fields
