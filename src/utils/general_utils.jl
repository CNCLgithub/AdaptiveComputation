export softmax,
    normalize_weights,
    findnearest_vec,
    retrieve_assignment,
    retrieve_preds,
    retrieve_obs,
    dist,
    find_distance_to_nd,
    findnearest,
    read_json


# stable softmax
function softmax(x)
    x = x .- maximum(x)
    return exp.(x) / sum(exp.(x))
end

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

"""
Retrieves assignment from trace for the given group

where group = :targets | :distractors
"""
function retrieve_assignment(trace, group)
    t, params = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    return choices[:chain => t => group]
end


"""
Retrieves the predicted target state for all targets
"""
function retrieve_preds(trace)
    t, _ = Gen.get_args(trace)
    targets = Gen.get_retval(trace)[end]

    positions = Array{Float64}(undef, length(targets), 2)
    for i=1:length(targets)
        positions[i,:] = [targets[i].x, targets[i].y]
    end
    return positions
end


"""
Retrieves the observation points for the specified time point
"""
function retrieve_obs(cm::Union{Gen.StaticChoiceMap,
                                Gen.DynamicChoiceMap}, t::Int)
    submap = Gen.get_submap(cm, :chain => t => :points)
    pos_t = Gen.to_array(submap,
                         Float64)
    pos = reshape(pos_t, 2, :)'
    return pos
end

function retrieve_obs(cm::Gen.ChoiceMap, t::Int)

    submap = Gen.get_submap(cm, :chain => t => :points)
    new_sm = Gen.DynamicChoiceMap(submap)
    pos_t = Gen.to_array(new_sm,
                         Float64)
    pos = reshape(pos_t, 2, :)'
    return pos
end

# computes distance between two points
function dist(x,y)
    return sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
end


# finds distance to nearest distractor from target
function find_distance_to_nd(target, distractors)
    distances = Array{Float64}(undef, size(distractors, 1))

    for d=1:size(distractors, 1)
        distractor = [distractors[d,1],distractors[d,2]]
        distances[d] = dist(target, distractor)
    end

    return minimum(distances)
end


# finds the index of the element in a (sorted vector) that is nearest to x
# https://discourse.julialang.org/t/findnearest-function/4143/3
# (taken from Dan)
function findnearest(a,x)
    length(a) > 0 || return 0:-1
    r = searchsorted(a,x)
    length(r) > 0 && return r
    last(r) < 1 && return searchsorted(a,a[first(r)])
    first(r) > length(a) && return searchsorted(a,a[last(r)])
    x-a[last(r)] < a[first(r)]-x && return searchsorted(a,a[last(r)])
    x-a[last(r)] > a[first(r)]-x && return searchsorted(a,a[first(r)])
    return first(searchsorted(a,a[last(r)])):last(searchsorted(a,a[first(r)]))
end

# cribbed from https://github.com/JuliaDynamics/DrWatson.jl/blob/abed4bc699d5c6049c6010a5bc78ca62149a3cc9/src/saving_tools.jl#L300-L307
function struct2dict(s)
    Dict(x => getfield(s, x) for x in fieldnames(typeof(s)))
end

"""
    read_json(path)
    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    open(path, "r") do f
        global data
        data = JSON.parse(f)
    end
    
    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end
