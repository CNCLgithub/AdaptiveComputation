export ExampleExperiment

@with_kw struct ExampleExperiment <: AbstractExperiment
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    k::Int = 50
end

get_name(::ExampleExperiment) = "example"

function run_inference(q::ExampleExperiment) # masks, init_positions, params)

    gm_params = load(GMMaskParams, q.gm)
    motion = load(BrownianDynamicsModel, q.motion)

    # generating positions
    init_positions, masks = dgp(q.k, gm_params, motion)
    
    # ######### testing with masks from the generative model
    # trace, _ = Gen.generate(gm_masks_static, (q.k, motion, gm_params))

    # masks = Vector{Vector{BitArray{2}}}(undef, q.k)
    # for t=1:q.k
        # println(size(trace[:states => t => :masks]))
        # masks[t] = trace[:states => t => :masks]
        # """
        # for i=1:gm_params.n_trackers
            # masks[t,i] = masks_t
        # end
        # """
    # end

    # init_positions = Array{Float64}(undef, gm_params.n_trackers, 3)
    # init_state, _ = Gen.get_retval(trace)
    # for i=1:gm_params.n_trackers
        # init_positions[i,:] = init_state.graph.elements[i].pos
    # end
    # ###########

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
        #cm[:states => t => :masks] = masks[t,:]
        cm[:states => t => :masks] = masks[t]
        observations[t] = cm
    end
    
    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_masks_static,
                                        (0, motion, gm_params),
                                        constraints,
                                        args,
                                        observations)

    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_state!,
                pop_stats = retrieve_confusability,
                stop_rejuv = early_stopping_confusability)


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


