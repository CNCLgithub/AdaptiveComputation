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
    p1::Tuple{Int, Int}
    p2::Tuple{Int, Int}
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
function get_mds_rf(rf::T,
                    mds::Vector{Matrix{Float64}},
                    prob_threshold) where {T <: AbstractReceptiveField}
    # cropping each mask according to the receptive field
    cropped_mds = map(md -> crop(rf, md), mds)
    # filtering masks that have at least one pixel above prob_threshold
    filter(md -> any(md .> prob_threshold), cropped_mds)
end


"""
    simple, nonhierarchical case
"""
function get_mask_distributions(objects, gm::GMMaskParams)
    pos = map(o->o.pos, objects)
    mask_args, trackers_img = get_masks_rvs_args(pos, gm)
    mask_distributions = map(first, mask_args)
end


"""
    gets
"""
function get_pmbrfs(rf::AbstractReceptiveField,
                    mds::Vector{Matrix{Float64}},
                    gm::AbstractGMParams) where {T <: AbstractReceptiveField}
    existence_prob = 0.99 # TODO remove constant?

    n = length(mds) + 1 # |mbrfs| + 1 for PPP
    
    # this is not completely correct, but maybe a fine approximation
    rf_n_pixels = reduce(*, get_dimensions(rf))
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
                     gm::AbstractGMParams) where T <: AbstractReceptiveField
    mds = get_mask_distributions(objects, gm)
    mds_rf = map(rf -> get_mds_rf(rf, mds, prob_threshold), rec_fields)
    rfs_vec = map(get_pmbrfs, rec_fields, mds_rf, fill(gm, length(rec_fields)))
end


###############
# A FEW UTILS for automatic generation of receptive
# fields given generative model params
###############
function get_rectangle_receptive_field(xy, n_x, n_y, gm)
    w = floor(Int, gm.img_width/n_y)
    h = floor(Int, gm.img_height/n_x)

    p1 = (w*(xy[1]-1)+1, h*(xy[2]-1)+1)
    p2 = p1 .+ (w-1, h-1)

    return RectangleReceptiveField(p1, p2)
end

"""
    get_rectangle_receptive_fields(n_x, n_y, gm)

    Arguments:
        n_x - number of receptive fields in the x dimension
        n_y - number of receptive fields in the y dimension
        gm - generative model parameters

"""
function get_rectangle_receptive_fields(n_x, n_y, gm)
    rf_idx = Iterators.product(1:n_x, 1:n_y)
    receptive_fields = map(xy -> get_rectangle_receptive_field(xy, n_x, n_y, gm), rf_idx)
    receptive_fields = map(i -> receptive_fields[i], 1:n_x*n_y) # I can't find a way to flatten ://///
end

function crop(rf::RectangleReceptiveField,
              mask_distribution::BitArray{2})
    idxs = CartesianIndices((rf.p1[1]:rf.p2[1], rf.p1[2]:rf.p2[2]))
    mask_distribution[idxs]
end

export RectangleReceptiveField, get_rectangle_receptive_fields
