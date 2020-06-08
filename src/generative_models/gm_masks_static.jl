export Tracker,
        gm_masks_static

struct Tracker
 	x::Float64
 	y::Float64
    z::Float64 # goes from 0 (closest) to 1 (fartherest)

 	vx::Float64
 	vy::Float64
end

struct FullState
    trackers::Vector{Tracker}
    pmbrfs_params::Union{PMBRFSParams, Nothing}
    #ppp_params::Union{PPPParams, Nothing}
    #mbrfs_params::Union{MBRFSParams, Nothing}
end


function get_masks_rvs_args(trackers, params)
    # sorted according to depth
    # (smallest values first, i.e. closest object first)
    
    # sorting trackers according to depth for rendering purposes
    depth_perm = sortperm(trackers[:, 3])
    trackers = trackers[depth_perm, :] 

    rvs_args = Vector{Tuple}(undef, length(trackers))

    # initially empty image
    #img_so_far = BitArray{2}(undef, params.img_height, params.img_width)
    #img_so_far .= false

    img_so_far = zeros(params.img_height, params.img_width)

    for i=1:params.num_trackers
        #mask = draw_mask(trackers[i,:], img_so_far, params)
        mask = draw_gaussian_mask(trackers[i,:], img_so_far, params)
        #img_so_far = img_so_far .| mask

        #mask_prob = fill(1e-6, params.img_height, params.img_width)
        #mask_prob[mask] .= 1.0 - 1e-6
        #push!(rvs_args, (mask_prob,))
        
        mask = subtract_images(mask, img_so_far)
        img_so_far = add_images(img_so_far, mask)
        rvs_args[i] = (mask,)
        #push!(rvs_args, (mask,))
    end
    
    # sorting arguments for MBRFS back so that tracker rvs_args[1] corresponds to tracker 1
    rvs_args = rvs_args[invperm(depth_perm)]

    return rvs_args, img_so_far
end


"""
    find_nearest_neighbour(distances::Matrix{Float64}, i::Int)
   
Returns the index of the nearest neighbour
"""
function find_nearest_neighbour(distances::Matrix{Float64}, i::Int)
    min_distance = Inf
    index = 0

    for j=1:size(distances,1)
        if j != i && min_distance > distances[i,j]
            min_distance = distances[i,j]
            index = j 
        end
    end

    return index
end

"""
    get_masks_params(trackers, params::Params)

Returns the masks parameters (for PMBRFS) - ppp_params, mbrfs_params
i.e. parameters for the pmbrfs random variable describing the masks
"""
function get_masks_params(trackers, params::Params)

    # compiling list of x,y,z coordinates of all objects
    objects = Matrix{Float64}(undef, params.num_trackers, 3)
    
    for i=1:length(trackers)
        objects[i,1] = trackers[i].x
        objects[i,2] = trackers[i].y
        objects[i,3] = trackers[i].z
    end
    
    distances = Matrix{Float64}(undef, params.num_trackers, params.num_trackers)
    for i=1:params.num_trackers
        for j=1:params.num_trackers
            a = [objects[i,1], objects[i,2]]
            b = [objects[j,1], objects[j,2]]
            distances[i,j] = dist(a, b)
        end
    end

    # probability of existence of a particular tracker in MBRFS masks set
    rs = zeros(params.num_trackers)
    scaling = 5.0 # parameter to tweak how close objects have to be to occlude
    missed_detection = 1e-30 # parameter to tweak probability of missed detection

    if params.num_trackers == 1
       rs = [1.0 - missed_detection]
    else
        for i=1:params.num_trackers
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
    
    rvs = fill(mask, params.num_trackers)
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

    ppp_params = PPPParams(params.num_distractors_rate, mask, mask_params)

    return ppp_params, mbrfs_params
end


##### INIT STATE ######
@gen function sample_init_tracker(params::Params)
    x = @trace(uniform(-250.0, 250.0), :x)
    y = @trace(uniform(-250.0, 250.0), :y)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    vx = @trace(normal(0.0, params.sigma_v), :vx)
    vy = @trace(normal(0.0, params.sigma_v), :vy)

    return Tracker(x, y, z, vx, vy)
end
init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(params)
    trackers_params = fill(params, params.num_trackers)
    init_trackers = @trace(init_trackers_map(trackers_params), :init_trackers)
    
    return FullState(init_trackers, nothing)
end
######################


##### UPDATE STATE #####
@gen function tracker_update_kernel(tracker::Tracker, params::Params)

    mu_vx = params.inertia * tracker.vx - params.spring * tracker.x
	vx = @trace(normal(mu_vx, params.sigma_w), :vx)

	mu_vy = params.inertia * tracker.vy - params.spring * tracker.y
	vy = @trace(normal(mu_vy, params.sigma_w), :vy)

	x = tracker.x + vx
	y = tracker.y + vy
    z = tracker.z

	new_tracker = Tracker(x, y, z, vx, vy)

	return new_tracker
end
trackers_update_map = Gen.Map(tracker_update_kernel)
##################################


@gen (static) function kernel(t::Int,
                     prev_state::FullState,
                     params::Params)

    prev_trackers = prev_state.trackers

    trackers_params = fill(params, params.num_trackers)
    new_trackers = @trace(trackers_update_map(prev_trackers, trackers_params), :trackers)
    
    # get masks params returns parameters for the poisson multi bernoulli
    ppp_params, mbrfs_params = get_masks_params(new_trackers, params)

    # initializing the saved state for the target designation
    pmbrfs_stats = PMBRFSStats([],[],[])
    pmbrfs_params = PMBRFSParams(ppp_params, mbrfs_params, pmbrfs_stats)

    @trace(pmbrfs(pmbrfs_params), :masks)

    # returning this to get target designation and assignment
    # later (HACKY STUFF) saving as part of state
    new_state = FullState(new_trackers, pmbrfs_params)

    return new_state
end

chain = Gen.Unfold(kernel)

@gen (static) function gm_masks_static(T::Int,
                                            params::Params)
    
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(chain(T, init_state, params), :states)

    result = (init_state, states)

    return result
end
