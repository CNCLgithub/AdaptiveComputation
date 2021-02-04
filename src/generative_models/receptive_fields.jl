abstract type AbstractReceptiveField end

using Combinatorics
using Statistics
using Images, FileIO, PaddedViews

abstract type AbstractRFParams end

@with_kw struct RectRFParams <: AbstractRFParams
    n_x::Int64 = 5
    n_y::Int64 = 5
    overlap::Int64 = 2
end

function load(::Type{RectRFParams}, path; kwargs...)
    RectRFParams(;read_json(path)..., kwargs...)
end
function load(::Type{RectRFParams}, path::String)
    RectRFParams(;read_json(path)...)
end

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
end

"""
    simple, nonhierarchical case
"""
function get_mask_distributions(objects, gm; flow_masks=nothing)
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
    existence_prob = 1.0 - 1e-10 # TODO remove constant?

    n = length(mds) + 1 # |mbrfs| + 1 for PPP
    
    # this is not completely correct, but maybe a fine approximation
    rf_n_pixels = prod(get_dimensions(rf))
    rf_proportion_of_img = rf_n_pixels/gm.img_width/gm.img_height
    rf_distractor_rate = gm.distractor_rate * rf_proportion_of_img

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

# returns true if the object position is within rf
function within(point, rf)
    x, y, _ = point
    x > rf.p1[1] && x < rf.p2[1] && y > rf.p1[2] && y < rf.p2[2]
end

#struct Gen.BroadcastedUniform <: Gen.Distribution{Array} end

function get_pmbrfs_points(rf, points, gm)
    existence_prob = 1.0 - 1e-10 # TODO remove constant?

    n = length(points) + 1 # |mbrfs| + 1 for PPP
    
    # this is not completely correct, but maybe a fine approximation
    rf_area = prod(get_dimensions(rf))
    rf_proportion = rf_area/gm.area_width/gm.area_height
    rf_distractor_rate = gm.distractor_rate * rf_proportion

    pmbrfs = RFSElements{Array}(undef, n)
    
    lows = [rf.p1[1], rf.p1[2], 0.0]
    highs = [rf.p2[1], rf.p2[2], 1.0]
    pmbrfs[1] = PoissonElement{Array}(rf_distractor_rate, broadcasted_uniform, (lows, highs))
    for i=2:n
        pmbrfs[i] = BernoulliElement{Array}(existence_prob, my_broadcasted_normal,
                                            (points[i-1], [gm.obs_noise, gm.obs_noise, 1.0]))
    end

    return pmbrfs
    
end

"""
    gets the vector of random finite sets for each receptive field
    for generative model that has point observations
"""
function get_rfs_vec_points(rec_fields::Vector{RectangleReceptiveField},
                            objects::Vector{Object},
                            gm)
    points = @>> objects map(x->x.pos)
    points_rf = @>> rec_fields map(rf -> filter(p -> within(p, rf), points))
    rfs_vec = map(get_pmbrfs_points, rec_fields, points_rf, fill(gm, length(rec_fields)))
    return rfs_vec
end


###############
# A FEW UTILS for automatic generation of receptive
# fields given generative model params
# and some other utils haha
###############

function bound(x, a, b)
    min(max(x, a), b)
end

function bound_point(p, w, h)
    (bound(p[1], 1, w), bound(p[2], 1, h))
end

function get_rectangle_receptive_field(xy, n_x, n_y, width, height;
                                       overlap = 0,
                                       point_observations = false)

    w = floor(Int, width/n_y)
    h = floor(Int, height/n_x)

    p1 = (w*(xy[1]-1)+1, h*(xy[2]-1)+1) .- (overlap, overlap)
    p1 = bound_point(p1, width, height)

    p2 = p1 .+ (w-1, h-1) .+ (overlap, overlap)
    p2 = bound_point(p2, width, height)
    
    # if observations are points, then shift fields so that
    # center of the whole area is at origin
    if point_observations
        shift = (-width/2.0, -height/2.0)
        p1 = p1 .+ shift
        p2 = p2 .+ shift
    end

    return RectangleReceptiveField(p1, p2) end

"""
    get_rectangle_receptive_fields(n_x, n_y, gm)

    Arguments:
        n_x - number of receptive fields in the x dimension
        n_y - number of receptive fields in the y dimension
        gm - generative model parameters

"""
function get_rectangle_receptive_fields(n_x, n_y, width, height;
                                        overlap = 0,
                                        point_observations = false)
    rf_idx = Iterators.product(1:n_x, 1:n_y)
    receptive_fields = map(xy -> get_rectangle_receptive_field(xy, n_x, n_y, width, height;
                                                               overlap=overlap,
                                                              point_observations = point_observations), rf_idx)
    receptive_fields = map(i -> receptive_fields[i], 1:n_x*n_y) # I can't find a way to flatten ://///
end

function crop(rf::RectangleReceptiveField,
              mask_bits::BitArray{2})
    idxs = CartesianIndices((rf.p1[1]:rf.p2[1], rf.p1[2]:rf.p2[2]))
    mask_bits[idxs]
    #mask_bits[1:32, 1:32]
end

function cropfilter(rf, masks)
    cropped_masks = map(mask -> MOT.crop(rf, mask), masks)
    croppedfiltered_masks = filter(mask -> any(mask .!= 0), cropped_masks)
end


### target designation utils

# gets indices of masks that fall into the receptive field
function cropindices(rf, masks)
    cropped_masks = map(m -> MOT.crop(rf, m), masks)
    indices = filter(i -> any(cropped_masks[i] .!= 0), 1:length(cropped_masks))
    @>> indices x -> (x, collect(1:length(x)))
end

"""
    gets the score for the particular target designation, e.g.
    td = [1,2,3,8]
    indices = 
    rf_assignment = 
"""
function get_td_score(td, indices, rf_assignment)
    # finding the intersecting global mask indices with each rf
    intersections_global = @>> indices map(idx -> intersect(idx[1], td))

    # mapping the intersections to the local level mask indices
    intersections_local = []
    for (i, intersection) in enumerate(intersections_global)
        intersection_indices = findall(x -> x in intersection, indices[i][1])
        push!(intersections_local, indices[i][2][intersection_indices])
    end
    
    td_score = 0.0
    for (i, intersection) in enumerate(intersections_local)
        dc = rf_assignment[i][1] # data correspondence
        scores = rf_assignment[i][2] # scores for data correspondence

        score = -Inf
        for j=1:length(dc)
            targets = unique(Iterators.flatten(dc[j][2:end]))
            if issetequal(targets, intersection)
                score = logsumexp(score, scores[j])
            end
        end

        td_score += score
    end
    return td_score
end

"""
    returns the target designation distribution
"""
function get_target_designation(n_targets,
                                receptive_field_assignment,
                                masks,
                                receptive_fields)
    indices = @>> receptive_fields map(rf -> cropindices(rf, masks))
    indices = reshape(indices, size(receptive_field_assignment))
   
    # all possible target designations
    tds = collect(combinations(1:length(masks), n_targets))
    scores = @>> tds map(td -> get_td_score(td, indices, receptive_field_assignment))
    perm = sortperm(scores, rev=true)
    @>> perm map(i -> (tds[i], scores[i]))
end

# concatenates masks from different receptive fields into one image
function concatenate(masks)
    @>> begin 1:size(masks, 1)
        map(i -> hcat(masks[i,:]...)) # concatenating horizontally
        h_masks->vcat(h_masks...) # concatenating vertically
    end
end

function pad_mask(mask; pad=1)
    h, w = size(mask)
    PaddedView(1, mask, (1:h+pad*2, 1:w+pad*2), (pad+1:h+pad, pad+1:w+pad))
end

function _extract_mask_distributions(rfs)
    mask_distributions = @>> rfs map(x -> first(x.args))
end

"""
    rfs_vec = shape(n_receptive_fields_x, n_receptive_fields_y)
"""
function save_receptive_fields_img(rfs_mat, t, out_dir)
    image = @>> begin rfs_mat
            map(rfs -> _extract_mask_distributions(rfs))
            map(x -> pad_mask.(x))
            map(rf -> mean(rf))
            concatenate
        end

    fn = joinpath(out_dir, "$(lpad(t,3,'0')).png")
    save(fn, image)
end

export RectangleReceptiveField, get_rectangle_receptive_fields, RectRFParams
