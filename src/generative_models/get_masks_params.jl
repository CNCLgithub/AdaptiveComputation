using LinearAlgebra

function get_masks_rvs_args(positions, params::AbstractGMParams)
    # sorted according to depth
    # (smallest values first, i.e. closest object first)
    
    # sorting trackers according to depth for rendering purposes
    depth_perm = sortperm(map(p->p[3], positions))
    positions = positions[depth_perm]

    rvs_args = Vector{Tuple}(undef, length(positions))
    
    # initially empty image
    img_so_far = zeros(params.img_height, params.img_width)
    
    # scaling the radius of the dot
    r = params.dot_radius/params.area_height*params.img_height

    for i=1:length(positions)
        x, y = translate_area_to_img(positions[i][1], positions[i][2],
                                       params.img_height, params.img_width,
                                       params.area_height, params.area_width,
                                       whole_number=false)
        
        mask = draw_gaussian_dot_mask([x,y], r,
                                 params.img_height, params.img_width,
                                 params.gauss_r_multiple,
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
function get_masks_params(positions, params::AbstractGMParams;
                          flow_masks=nothing)
    
    # compiling list of x,y,z coordinates of all objects
    #objects = [trackers[i].pos[j] for i=1:params.n_trackers, j=1:3]
    distances = [norm(positions[i][1:2] - positions[j][1:2])
                 for i=1:length(positions), j=1:length(positions)]
    
    # let's assume all dots exist
    rs = ones(length(positions))

    mask_args, trackers_img = get_masks_rvs_args(positions, params)

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
    
    pmbrfs = RFSElements{Array}(undef, length(positions) + 1)
    pmbrfs[1] = PoissonElement{Array}(params.distractor_rate, mask, (clutter_mask,))
    for i = 2:length(pmbrfs)
        idx = i - 1
        pmbrfs[i] = BernoulliElement{Array}(rs[idx], mask, mask_args[idx])
    end
    pmbrfs, flow_masks
end
