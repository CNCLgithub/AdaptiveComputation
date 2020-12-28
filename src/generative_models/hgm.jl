export hgm_pos,
        HGMParams,
        default_hgm

using LinearAlgebra

@with_kw struct HGMParams <: AbstractGMParams
    n_trackers::Int = 2
    distractor_rate::Real = 2.0
    init_pos_spread::Real = 300.0
    polygon_radius::Real = 80.0
    
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

function load(::Type{HGMParams}, path; kwargs...)
    HGMParams(;read_json(path)..., kwargs...)
end

function load(::Type{HGMParams}, path::String)
    HGMParams(;read_json(path)...)
end

const default_hgm = HGMParams()

# TODO write a hierarchical version of this
function get_masks_rvs_args(trackers, params::HGMParams)
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
    get_masks_params(trackers, params::Params)
Returns the masks parameters (for PMBRFS) - ppp_params, mbrfs_params
i.e. parameters for the pmbrfs random variable describing the masks
"""
function get_masks_params(trackers, params::GMMaskParams)

    # compiling list of x,y,z coordinates of all objects
    objects = [trackers[i].pos[j] for i=1:params.n_trackers, j=1:3]
    distances = [norm(objects[i,1:2] - objects[j,1:2])
                for i=1:params.n_trackers, j=1:params.n_trackers]
    
    # probability of existence of a particular tracker in MBRFS masks set
    rs = zeros(params.n_trackers)
    scaling = 5.0 # parameter to tweak how close objects have to be to occlude
    missed_detection = 1e-30 # parameter to tweak probability of missed detection

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

    rvs = fill(MOT.mask, params.n_trackers)
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
@gen function sample_init_dot_or_polygon(gmh)::Object
    
    x = @trace(uniform(-gmh.init_pos_spread, gmh.init_pos_spread), :x)
    y = @trace(uniform(-gmh.init_pos_spread, gmh.init_pos_spread), :y)

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)
    
    # sampling whether we're dealing with a polygon :O
    pol = @trace(bernoulli(0.5), :polygon)
    if pol
        # 3, 4 or 5 dots in the polygon
        n_dots = Gen.categorical(fill(1.0/3, 3)) + 2
        
        dots = Vector{Dot}(undef, n_dots)
        for i=1:n_dots
            # creating dots along the polygon
            r = gmh.polygon_radius
            dot_x = x + r * cos(2*pi*i/n_dots)
            dot_y = y + r * sin(2*pi*i/n_dots)
            
            # sprinkling some noise
            dot_x = @trace(normal(dot_x, 5.0), i => :x)
            dot_y = @trace(normal(dot_y, 5.0), i => :y)

            dots[i] = Dot([dot_x,dot_y,z], [0.0,0.0])
        end

        return Polygon([x,y,z], [0.0,0.0], gmh.polygon_radius, dots)
    else
        return Dot([x,y,z], [0.0,0.0])
    end
end

@gen function sample_init_hierarchical_state(hgm::HGMParams)::Object
    trackers_gm = fill(gm.init_pos_spread, gm.n_trackers)
    exp0 = fill(gm.exp0, gm.n_trackers)
    trackers = @trace(init_dot_or_polygon_map(trackers_gm, exp0), :trackers)
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

init_dot_or_polygon_map = Gen.Map(sample_init_dot_or_polygon)

@gen function sample_init_hierarchical_state(hgm::HGMParams)
    hgm_trackers = fill(hgm, hgm.n_trackers)
    trackers = @trace(init_dot_or_polygon_map(hgm_trackers), :trackers)
    trackers = collect(Object, trackers)
    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)

    pmbrfs = RFSElements{Array}(undef, 0)

    if hgm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, hgm.n_trackers, hgm.fmasks_n)
        for i=1:hgm.n_trackers
            for j=1:hgm.fmasks_n
                fmasks[i,j] = zeros(hgm.img_height, hgm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               hgm.fmasks_decay_function)
    else
        flow_masks = nothing
    end

    FullState(graph, pmbrfs, flow_masks)

    return FullState(graph, pmbrfs, flow_masks)
end


@gen function hgm_pos_kernel(t::Int,
                            prev_state::FullState,
                            dynamics_model::AbstractDynamicsModel,
                            params::GMMaskParams)
    prev_graph = prev_state.graph
    new_graph = @trace(hgm_update(dynamics_model, prev_graph), :dynamics)
    new_trackers = new_graph.elements
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = FullState(new_graph, pmbrfs, nothing)
    return new_state
end

hgm_pos_chain = Gen.Unfold(hgm_pos_kernel)

@gen function hgm_pos(k::Int, motion::AbstractDynamicsModel,
                      params::HGMParams)
    
    init_state = @trace(sample_init_hierarchical_state(params), :init_state)
    states = @trace(hgm_pos_chain(k, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end
