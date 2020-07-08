export Exp0MaskRCNN

@with_kw struct Exp0MaskRCNN <: AbstractExperiment
    trial::Int = 1
    dataset_path::String = "output/datasets/exp_0.h5"
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
end

get_name(::Exp0MaskRCNN) = "exp0_mask_rcnn"

function run_inference(q::Exp0MaskRCNN)

    gm_params = load(GMMaskParams, q.gm)
    
    # generating initial positions and masks (observations)
    init_positions, true_masks, motion, positions = load_exp0_trial(q.trial, gm_params, q.dataset_path)
    positions = positions[1:q.k]
    
    # rendering images in memory (array=true)
    imgs = render(positions, gm_params,
                  stimuli=true,
                  array=true)
    
    println("getting those masks from Mask RCNN...")
    masks = Vector{Vector{BitArray{2}}}(undef, q.k)
    for t=1:q.k-1
        print("timestep: $t / $(q.k) \r")
        masks_t = []
        chan_img = channelview(RGB.(imgs[t]))
        masks_bool = mask_rcnn.get_masks(chan_img)
        for i=1:size(masks_bool,1)
            mask_bool = transpose(masks_bool[i,:,:])
            mask_bool = imresize(mask_bool, gm_params.img_height, gm_params.img_width)
            mask_bool = round.(Int, mask_bool)
            push!(masks_t, BitArray(mask_bool))
        end
        masks[t] = masks_t
    end
    println("Mask RCNN done!")
    
    # last timestep from true masks,
    # so that we know performance
    masks[q.k] = true_masks[q.k]

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                                :assignments => extract_assignments,
                                :tracker_masks => extract_tracker_masks
                               ))

    
    # initial observations based on init_positions
    # model knows where trackers start off
    constraints = Gen.choicemap()
    for i=1:size(init_positions, 1)
        addr = :init_state => :trackers => i => :x
        constraints[addr] = init_positions[i,1]
        addr = :init_state => :trackers => i => :y
        constraints[addr] = init_positions[i,2]
    end
    
    # compiling further observations for the model
    args = [(t, motion, gm_params) for t in 1:q.k]
    observations = Vector{Gen.ChoiceMap}(undef, q.k)
    for t = 1:q.k
        cm = Gen.choicemap()
        cm[:states => t => :masks] = masks[t]
        observations[t] = cm
    end
    
    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_masks_static,
                                        (0, motion, gm_params),
                                        constraints,
                                        args,
                                        observations)

    attention = load(TDEntropyAttentionModel, q.attention;
                     perturb_function = perturb_state!)

    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))
    

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = nothing)
    

    extracted = extract_chain(results)
    tracker_positions = extracted["unweighted"][:tracker_positions]
    tracker_masks = extracted["unweighted"][:tracker_masks]
    
    aux_state = extracted["aux_state"]
    attempts = Vector{Int}(undef, q.k)
    attended = Vector{Vector{Float64}}(undef, q.k)
    for t=1:q.k
        attempts[t] = aux_state[t].attempts
        attended[t] = aux_state[t].attended_trackers
    end
    
    plot_attention(attended)
    
    # visualizing inference on stimuli
    render(positions, gm_params;
           pf_xy=tracker_positions[:,:,:,1:2],
           attended=attended/attention.max_sweeps,
           tracker_masks=tracker_masks)

    # visualizing inference
    full_imgs = get_full_imgs(masks)
    visualize(tracker_positions[:,:,:,1:2], full_imgs, gm_params, "inference_render")

    return results
end


