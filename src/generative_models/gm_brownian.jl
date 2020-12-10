using LinearAlgebra

struct FullState
    graph::CausalGraph{Object, SimpleGraph}
    rfs::RFSElements{Array}
    flow_masks::Union{Nothing, FlowMasks}
end

@with_kw struct GMMaskParams
    n_trackers::Int = 4
    distractor_rate::Real = 4.0
    init_pos_spread::Real = 300.0
    
    # in case of BDot and CBM
    init_vel::Real = 5.0
    
    # graphics parameters
    dot_radius::Real = 20.0
    img_height::Int = 200
    img_width::Int = 200
    area_height::Int = 800
    area_width::Int = 800

    # parameters for the drawing the mask random variable arguments
    dot_p::Float64 = 0.5 # prob of pixel on in the dot region
    gauss_amp::Float64 = 0.5 # gaussian amplitude for the gaussian component of the mask
    gauss_std::Float64 = 2.5 # standard deviation --||--

    # rfs parameters
    record_size::Int = 100 # number of associations

    # legacy support for exp0
    exp0::Bool = false

    # flow masks
    fmasks::Bool = false
    fmasks_decay_function::Function = default_decay_function
    fmasks_n = 5

    # probes
    probe_flip::Float64 = 0.0
end

function load(::Type{GMMaskParams}, path; kwargs...)
    GMMaskParams(;read_json(path)..., kwargs...)
end

const default_gm = GMMaskParams()

function load(::Type{GMMaskParams}, path::String)
    GMMaskParams(;read_json(path)...)
end

function get_masks_rvs_args(trackers, params::GMMaskParams)
    # sorted according to depth
    # (smallest values first, i.e. closest object first)
    
    # sorting trackers according to depth for rendering purposes
    depth_perm = sortperm(trackers[:, 3])
    trackers = trackers[depth_perm, :]

    rvs_args = Vector{Tuple}(undef, params.n_trackers)
    
    # initially empty image
    img_so_far = zeros(params.img_height, params.img_width)
    
    # scaling the radius of the dot
    r = params.dot_radius/params.area_height*params.img_height

    for i=1:size(trackers, 1)
        x, y = translate_area_to_img(trackers[i,1], trackers[i,2],
                                       params.img_height, params.img_width,
                                       params.area_height, params.area_width,
                                       whole_number=false)
        
        mask = draw_gaussian_dot_mask([x,y], r,
                                 params.img_height, params.img_width,
                                 params.dot_p,
                                 params.gauss_amp, params.gauss_std)
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
function get_masks_params(trackers, params::GMMaskParams;
                          flow_masks=nothing)

    # compiling list of x,y,z coordinates of all objects
    objects = [trackers[i].pos[j] for i=1:params.n_trackers, j=1:3]
    distances = [norm(objects[i,1:2] - objects[j,1:2])
                for i=1:params.n_trackers, j=1:params.n_trackers]
    
    # probability of existence of a particular tracker in MBRFS masks set
    rs = zeros(params.n_trackers)
    scaling = 5.0 # parameter to tweak how close objects have to be to occlude
    missed_detection = 1e-30 # parameter to tweak probability of missed detection

    # legacy support for exp0 - masks always present
    if params.exp0
        rs = ones(params.n_trackers)
    else
        if params.n_trackers == 1
            rs = [1.0 - missed_detection]
        else
            for i=1:params.n_trackers
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
    end
    
    rs = ones(params.n_trackers) # CHANGED

    mask_args, trackers_img = get_masks_rvs_args(objects, params)

    # explaining distractor with one uniform mask with trackers cutout
    # probability of sampling true on individual pixel given that one distractor is present
    pixel_prob = (params.dot_radius*pi^2)/(params.img_width*params.img_height)
    # getting this in the array with size of the image
    mask_prob = fill(pixel_prob, (params.img_height, params.img_width))
    clutter_mask = subtract_images(mask_prob, trackers_img)
    
    #display(flow_masks)
    if !isnothing(flow_masks)
        flow_masks, mask_args = add_flow_masks(flow_masks, mask_args)
    end
    
    pmbrfs = RFSElements{Array}(undef, params.n_trackers + 1)
    pmbrfs[1] = PoissonElement{Array}(params.distractor_rate, mask, (clutter_mask,))
    for i = 2:length(pmbrfs)
        idx = i - 1
        pmbrfs[i] = BernoulliElement{Array}(rs[idx], mask, mask_args[idx])
    end
    pmbrfs, flow_masks
end


# ##### INIT STATE ######
# #@gen (static) function sample_init_tracker(init_pos_spread::Real)::Dot
@gen function sample_init_tracker(init_pos_spread::Real, exp0::Bool)::Dot
    
    # legacy support for exp0
    if exp0
        # exp0 initial position was sampled from normal (0.0, 30.0)
        x = @trace(normal(0.0, 30.0), :x)
        y = @trace(normal(0.0, 30.0), :y)
        # exp0 initial velocity was sampled from normal(0.0, 2.2)
        vx = @trace(normal(0.0, 2.2), :vx)
        vy = @trace(normal(0.0, 2.2), :vy)
    else
        x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
        y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)
        # initial velocity is zero (in the new version)
        vx = 0.0
        vy = 0.0
    end

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    # initial velocity is zero
    return Dot([x,y,z], [vx, vy])
end

init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(gm::GMMaskParams)
    trackers_gm = fill(gm.init_pos_spread, gm.n_trackers)
    exp0 = fill(gm.exp0, gm.n_trackers)
    trackers = @trace(init_trackers_map(trackers_gm, exp0), :trackers)
    trackers = collect(Dot, trackers)
    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)
    pmbrfs = RFSElements{Array}(undef, 0)

    if gm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, gm.n_trackers, gm.fmasks_n)
        for i=1:gm.n_trackers
            for j=1:gm.fmasks_n
                fmasks[i,j] = zeros(gm.img_height, gm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               gm.fmasks_decay_function)
    else
        flow_masks = nothing
    end

    FullState(graph, pmbrfs, flow_masks)
end


##################################
@gen static function br_pos_kernel(t::Int,
                            prev_state::FullState,
                            dynamics_model::AbstractDynamicsModel,
                            params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(brownian_update(dynamics_model, prev_graph), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end

br_pos_chain = Gen.Unfold(br_pos_kernel)

@gen static function gm_brownian_pos(T::Int, motion::AbstractDynamicsModel,
                                       params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(br_pos_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end

@gen static function br_mask_kernel(t::Int,
                            prev_state::FullState,
                            dynamics_model::AbstractDynamicsModel,
                            params::GMMaskParams)

    prev_graph = prev_state.graph

    new_graph = @trace(brownian_update(dynamics_model, prev_graph), :dynamics)
    new_trackers = new_graph.elements

    pmbrfs, flow_masks = get_masks_params(new_trackers, params, flow_masks=prev_state.flow_masks)
    @trace(rfs(pmbrfs), :masks)

    # returning this to get target designation and assignment
    new_state = FullState(new_graph, pmbrfs, flow_masks)

    return new_state
end

br_mask_chain = Gen.Unfold(br_mask_kernel)

@gen static function gm_brownian_mask(T::Int, motion::AbstractDynamicsModel,
#@gen function gm_brownian_mask(T::Int, motion::AbstractDynamicsModel,
                                       params::GMMaskParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(br_mask_chain(T, init_state, motion, params), :kernel)

    result = (init_state, states)

    return result
end

export GMMaskParams, gm_brownian_pos, gm_brownian_mask, default_gm
