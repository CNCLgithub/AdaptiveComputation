export load_exp0_trial


"""
loads init_positions, masks and the motion model from exp0 dataset
"""
function load_exp0_trial(trial, gm, dataset_path;
                         generate_masks=true,
                         from_mask_rcnn=false)

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
    positions = Vector{Array{Float64}}(undef, k)

    for t=1:k
        n_dots = size(dots[t,:,:], 1)
        positions[t] = Array{Float64}(undef, n_dots, 3)
         
        for i=1:n_dots
            positions[t][i,1:2] = dots[t,i,:]
            positions[t][i,3] = uniform(0,1)
        end
    end
    
    if generate_masks
        if from_mask_rcnn
            # rendering images in memory (array=true)
            imgs = render(positions, gm,
                          stimuli=true,
                          array=true)
        
            println("getting those masks from Mask RCNN...")

            masks = Vector{Vector{BitArray{2}}}(undef, k)
            for t=1:k-1
                print("timestep: $t / $(k) \r")
                masks_t = []
                chan_img = channelview(RGB.(imgs[t]))
                masks_bool = mask_rcnn.get_masks(chan_img)
                for i=1:size(masks_bool,1)
                    mask_bool = transpose(masks_bool[i,:,:])
                    mask_bool = imresize(mask_bool, gm.img_height, gm.img_width)
                    mask_bool = round.(Int, mask_bool)
                    push!(masks_t, BitArray(mask_bool))
                end
                masks[t] = masks_t
            end

            println("Mask RCNN done!")
            # getting the last masks from ground truth
            masks[k] = get_masks(positions, gm.dot_radius,
                                 gm.img_height, gm.img_width,
                                 gm.area_height, gm.area_width)[k]
        else
            masks = get_masks(positions,
                              gm.dot_radius,
                              gm.img_height, gm.img_width,
                              gm.area_height, gm.area_width)
        end
    else
        masks = nothing
    end

    # getting the motion model
	inertia = data["inertia"]
	spring = data["spring"]
	sigma_w = data["sigma_w"]
    motion = BrownianDynamicsModel(inertia, spring, sigma_w, sigma_w)
    
    return init_positions, masks, motion, positions
end
