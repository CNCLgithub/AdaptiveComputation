abstract type AbstractReceptiveField end

"""
    parametrized by two points:
    p1 = (xmin, ymin)
    p2 = (xmax, ymax)
"""
struct RectangleReceptiveField <: AbstractReceptiveField
    p1::Tuple{Int, Int}
    p2::Tuple{Int, Int}
end

function get_bool_representation(::T, ::G) where {T <:AbstractReceptiveField, G <: AbstractGMParams}
    println("not implemented")
end

function _in(x, a, b)
    x > a & x < b
end

# TODO not used? -> delete
function get_bool_representation(rf::RectangleReceptiveField,
                                 gm::G) where {G <: AbstractGMParams}
    xys = Iterators.product(1:gm.img_height, 1:gm.img_width)
    br = map(xy -> _in(xy[1], rf.p1[1], rf.p2[1]) & _in(xy[2], rf.p1[2], rf.p2[2]), xys)
end


function crop(rf::T,
              mask_distribution::Matrix{Float64}) where {T <: AbstractReceptiveField}
    println("not implemented")
end

function crop(rf::RectangleReceptiveField,
              mask_distribution::Matrix{Float64})
    idxs = CartesianIndices((rf.p1[1]:rf.p2[1], rf.p1[2]:rf.p2[2]))
    mask_distribution[idxs]
end


# gets the mask distributions for a given receptive field
function get_mds_rf(rf::T,
                    mds::Vector{Matrix{Float64}},
                    prob_threshold) where {T <: AbstractReceptiveField}
    
    cropped_mds = map(md -> crop(rf, md), mds)
    mds_rf = filter(md -> any(md .> prob_threshold), cropped_mds)
    display(mds_rf)
    return mds_rf
end


"""
    simple, nonhierarchical case
"""
function get_mask_distributions(objects, gm::GMMaskParams)
    pos = map(o->o.pos, objects)
    mask_args, trackers_img = get_masks_rvs_args(pos, gm)
    mask_distributions = map(first, mask_args)
end

function get_dimensions(::T) where T <: AbstractReceptiveField
    println("not implemented")
end

function get_dimensions(rf::RectangleReceptiveField)
    (w, h) = rf.p2 .- rf.p1 .+ (1, 1)
end

"""
    gets
"""
function get_pmbrfs(rf::AbstractReceptiveField,
                    mds::Vector{Matrix{Float64}},
                    gm::AbstractGMParams) where {T <: AbstractReceptiveField}
    existence_prob = 0.99 # TODO remove constant?
    n = length(mds) + 1 # |mbrfs| + 1 for PPP
    
    # TODO fix clutter mask with proper probability
    # and potentially subtract tracker img
    clutter_mask = fill(0.01, get_dimensions(rf)...)
    
    """
    pmbrfs = RFSElements{Array}(undef, n)
    pmbrfs[1] = PoissonElement{Array}(gm.distractor_rate, mask, (clutter_mask,))
    for i=2:n
        pmbrfs[i] = BernoulliElement{Array}(existence_prob, mask, (mds[i-1],))
    end
    """

    if n==1
        # this is for testing purposes, zero prob mask to have at least one element in rfs
        pmbrfs = RFSElements{Array}(undef, 1)
        pmbrfs[1] = BernoulliElement{Array}(0, mask, (rand(get_dimensions(rf)...),))
    else
        pmbrfs = RFSElements{Array}(undef, n-1)
        for i=1:n-1
            pmbrfs[i] = BernoulliElement{Array}(existence_prob, mask, (mds[i],))
        end
    end

    display(pmbrfs)
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

export RectangleReceptiveField
