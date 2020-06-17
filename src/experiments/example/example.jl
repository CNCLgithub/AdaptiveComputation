export ExampleExperiment

@with_kw struct ExampleExperiment <: AbstractExperiment
    proc::String = "src/experiments/example/proc.json"
    gm::String = "src/experiments/example/gm.json"
    motion::String = "src/experiments/example/motion.json"
    k::Int = 20
end

get_name(::ExampleExperiment) = "example"

function run_inference(q::ExampleExperiment) # masks, init_positions, params)

    gm_params = load(GMMaskParams, q.gm)
    motion = load(BrownianDynamicsModel, q.motion)

    # generating positions
    init_positions, masks = dgp(q.k, gm_params, motion)


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
    args = [(t, gm_params) for t in 1:k]
    observations = Vector{Gen.ChoiceMap}(undef, q.k)
    for t = 1:k
        cm = Gen.choicemap()
        cm[:states => t => :masks] = masks[t,:]
        observations[t] = cm
    end
    
    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_masks_static,
                                        (0, gm_params),
                                        constraints,
                                        args,
                                        observations)

    proc = load(PopParticleFilter, q.proc;
                rejuvination = foo,
                pop_stats = bar,
                stop_rejuv = bamf)


    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = nothing)

    extracted = extract_chain(results)
    tracker_positions = extracted["unweighted"][:tracker_positions]

    # getting the images
    full_imgs = get_full_imgs(masks)

    # this is visualizing what the observations look like (and inferred state too)
    # you can find images under full_imgs/
    visualize(tracker_positions, full_imgs, gm_params)
    return results
end


