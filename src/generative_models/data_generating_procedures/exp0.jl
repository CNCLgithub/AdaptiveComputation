export load_exp0_trial


"""
loads init_positions, masks and the motion model from exp0 dataset
"""
function load_exp0_trial(trial, gm, dataset_path)

	file = h5open(dataset_path, "r")
	dataset = read(file, "dataset")
	data = dataset["$(trial-1)"]
    
    # getting initial positions
	init_dots = data["init_dots"]
    init_positions = Array{Float64}(undef, gm.n_trackers, 3)
    for i=1:gm.n_trackers
        init_positions[i,1:2] = init_dots[i,:]
        init_positions[i,3] = 0.5
    end

    # getting masks
	dots = data["gt_dots"]
    k = size(dots, 1)
    n_dots = Int(gm.n_trackers + gm.distractor_rate)
    positions = Array{Float64}(undef, k, n_dots, 3)

    # sampling a depth ordering
    depth = Vector{Float64}(undef, n_dots)
    for i=1:n_dots
        depth[i] = uniform(0,1)
    end

    for t=1:k
        for i=1:n_dots
            positions[t,i,1:2] = dots[t,i,:]
            positions[t,i,3] = 0.5 #depth[i]
        end
    end

    masks = get_masks(positions, gm.dot_radius,
                      gm.img_height, gm.img_width,
                      gm.area_height, gm.area_width)

    # getting the motion model
	inertia = data["inertia"]
	spring = data["spring"]
	sigma_w = data["sigma_w"]
    motion = BrownianDynamicsModel(inertia, spring, sigma_w, sigma_w)
    
    return init_positions, masks, motion
end
