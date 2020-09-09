export load_exp0_trial

using HDF5

"""
loads init_positions, masks and the motion model from exp0 dataset
"""
function load_exp0_trial(trial, gm, dataset_path;
                         generate_masks=true,
                         from_mask_rcnn=false)

	file = h5open(dataset_path, "r")
	dataset = read(file, "dataset")
	data = dataset["$(trial-1)"]

    n_dots = Int(gm.n_trackers + gm.distractor_rate)
    zs = [uniform(0,1) for i=1:n_dots] # sampling zs at the beginning
    
    # getting initial positions
	init_dots = data["init_dots"]
    init_positions = Array{Float64}(undef, gm.n_trackers, 3)
    for i=1:gm.n_trackers
        init_positions[i,1:2] = init_dots[i,:]
        init_positions[i,3] = zs[i] # CHANGED
    end

    # getting masks
	positions = data["gt_dots"]
    k = size(positions, 1)
    gt_causal_graphs = Vector{CausalGraph}(undef, k)

    # positions = Vector{Array{Float64}}(undef, k)

    for t=1:k
        # n_dots = size(positions[t,:,:], 1)
        # positions[t] = Array{Float64}(undef, n_dots, 3)
        dots = Vector{Dot}(undef, n_dots)
         
        for i=1:n_dots
            # positions[t][i,1:2] = positions[t,i,:]
            # positions[t][i,3] = zs[i] # CHANGED
            position = Vector{Float64}(undef, 3)
            position[1:2] = positions[t,i,:]
            position[3] = zs[i]
            dots[i] = Dot(position, zeros(2))
        end
        gt_causal_graphs[t] = CausalGraph(dots, SimpleGraph)
    end

    if generate_masks
        masks = get_masks(gt_causal_graphs, gm)
        if from_mask_rcnn
            masks[1:k-1] = get_masks_from_mask_rcnn(positions, gm)[1:k-1]
        end
    else
        masks = nothing
    end

    # getting the motion model
	inertia = data["inertia"]
	spring = data["spring"]
	sigma_w = data["sigma_w"]
    motion = BrownianDynamicsModel(inertia, spring, sigma_w, sigma_w)
    
    trial_data = Dict([:init_positions => init_positions,
                       :motion => motion,
                       :masks => masks,
                       :gt_causal_graphs => gt_causal_graphs])
end
