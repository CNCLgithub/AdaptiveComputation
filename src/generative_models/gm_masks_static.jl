export gm_masks_static

using LinearAlgebra

struct FullState
    graph::CausalGraph{Dot, SimpleGraph}
    pmbrfs_params::Union{PMBRFSParams, Nothing}
end

@with_kw struct GMMaskParams
    n_trackers::Int = 1
    distractor_rate::Real = 0.0
    init_pos_spread::Real = 400.0
    dot_radius::Real = 20.0
    img_height::Int = 200
    img_width::Int = 200
    area_height::Int = 800
    area_width::Int = 800
end

function load(::Type{GMMaskParams}, path::String)
    GMMaskParams(;read_json(path)...)
end

function get_masks_rvs_args(trackers, params::GMMaskParams)
    # sorted according to depth
    # (smallest values first, i.e. closest object first)
    
    # sorting trackers according to depth for rendering purposes
    depth_perm = sortperm(trackers[:, 3])
    trackers = trackers[depth_perm, :] 

    rvs_args = Vector{Tuple}(undef, length(trackers))
    
    # initially empty image
    img_so_far = zeros(params.img_height, params.img_width)

    for i=1:size(trackers, 1)
        mask = draw_gaussian_dot(trackers[i,1:2], params.dot_radius,
                                 params.img_height, params.img_width)
        mask = subtract_images(mask, img_so_far)
        img_so_far = add_images(img_so_far, mask)

        rvs_args[i] = (mask,)
    end
    
    # sorting arguments for MBRFS back so that tracker rvs_args[1] corresponds to tracker 1
    rvs_args = rvs_args[invperm(depth_perm)]

    return rvs_args, img_so_far
end


"""
    find_nearest_neighbour(distances::Matrix{Float64}, i::Int)
   
Returns the index of the nearest neighbour
"""
# TODO the copy statement is inefficient
function find_nearest_neighbour(distances::Matrix{Float64}, i::Int)
    d = copy(distances[i,:])
    d[i] = Inf
    return argmin(d)
end

"""
    get_masks_params(trackers, params::Params)

Returns the masks parameters (for PMBRFS) - ppp_params, mbrfs_params
i.e. parameters for the pmbrfs random variable describing the masks
"""
function get_masks_params(trackers, params::GMMaskParams)
    
    num_trackers = length(trackers)

    # compiling list of x,y,z coordinates of all objects
    objects = [trackers[i].pos[j] for i=1:num_trackers, j=1:3]
    distances = [norm(objects[i,1:2] - objects[j,1:2])
                for i=1:num_trackers, j=1:num_trackers]
    
    # probability of existence of a particular tracker in MBRFS masks set
    rs = zeros(num_trackers)
    scaling = 5.0 # parameter to tweak how close objects have to be to occlude
    missed_detection = 1e-30 # parameter to tweak probability of missed detection

    if num_trackers == 1
       rs = [1.0 - missed_detection]
    else
        for i=1:num_trackers
            j = find_nearest_neighbour(distances, i)
            
            # comparing the depth
            if objects[i,3] > objects[j,3]
                rs[i] = 1.0 - missed_detection
            else
                r = 1.0 - exp(-distances[i,j] * scaling)
                r -= missed_detection
                rs[i] = max(r, 0.0) # lower bound 0.0
            end
        end
    end
    
    rvs = fill(mask, num_trackers)
    rvs_args, trackers_img = get_masks_rvs_args(objects, params)
    mbrfs_params = MBRFSParams(rs, rvs, rvs_args)

    # explaining distractor with one uniform mask with trackers cutout
    # probability of sampling true on individual pixel given that one distractor is present
    pixel_prob = (params.dot_radius*pi^2)/(params.img_width*params.img_height)
    # getting this in the array with size of the image
    mask_prob = fill(pixel_prob, (params.img_height, params.img_width))
    #mask_prob[trackers_img] .= 1e-6
    mask_prob = subtract_images(mask_prob, trackers_img)
    mask_params = (mask_prob,)

    ppp_params = PPPParams(params.distractor_rate, mask, mask_params)

    return ppp_params, mbrfs_params
end


##### INIT STATE ######
@gen (static) function sample_init_tracker(init_pos_spread::Real)::Dot
    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)
    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)
    # initial velocity is zero
    return Dot([x,y,z], [0,0])
end

init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(params::GMMaskParams)
    trackers_params = fill(params.init_pos_spread, params.n_trackers)
    trackers = @trace(init_trackers_map(trackers_params), :trackers)
    trackers = collect(Dot, trackers)
    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)
    return FullState(graph, nothing)
end


##################################


#@gen (static) function kernel(t::Int,
@gen function kernel(t::Int,
                     prev_state::FullState,
                     dynamics_model::BrownianDynamicsModel,
                     params::GMMaskParams)

    prev_graph = prev_state.graph

    new_graph = @trace(brownian_update(dynamics_model, prev_graph), :dynamics)
    println(typeof(new_graph))
    new_trackers = new_graph.elements

    # get masks params returns parameters for the poisson multi bernoulli
    ppp_params, mbrfs_params = get_masks_params(new_trackers, params)

    # initializing the saved state for the target designation
    pmbrfs_stats = PMBRFSStats([],[],[])
    pmbrfs_params = PMBRFSParams(ppp_params, mbrfs_params, pmbrfs_stats)

    @trace(pmbrfs(pmbrfs_params), :masks)

    # returning this to get target designation and assignment
    # later (HACKY STUFF) saving as part of state
    new_state = FullState(new_graph, pmbrfs_params)

    return new_state
end

chain = Gen.Unfold(kernel)

#@gen (static) function gm_masks_static(T::Int, motion::AbstractDynamicsModel,
@gen function gm_masks_static(T::Int, motion::AbstractDynamicsModel,
                                       params::GMMaskParams)
    
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(chain(T, init_state, motion, params), :states)

    result = (init_state, states)

    return result
end
