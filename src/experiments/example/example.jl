export ExampleExperiment


@with_kw struct ExampleExperiment <: AbstractExperiment
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
end

get_name(::ExampleExperiment) = "example"

function run_inference(q::ExampleExperiment, path::String)

    gm_params = load(GMMaskParams, q.gm)
    motion = load(BrownianDynamicsModel, q.motion)
    
    # generating initial positions and masks (observations)
    init_positions, init_vels, masks, positions = dgp(q.k, gm_params, motion)

    # testing less inertia in dynamics for inference
    #motion = @set motion.inertia = 0.99
    #motion = @set motion.spring = 0.001
    #motion = @set motion.sigma_w = 2.5

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
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

    # getting the images
    full_imgs = get_full_imgs(masks)

    # this is visualizing what the observations look like (and inferred state too)
    # you can find images under inference_render
    visualize(tracker_positions, full_imgs, gm_params)
    

    return results
end


