export run_inference


function run_inference(masks, init_positions, params)

    T = size(masks, 1)

    println("preparing for inference...")

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                               ))

    
    # initial observations based on init_positions
    # model knows where trackers start off
    init_obs = Gen.choicemap()
    for i=1:params["inference_params"]["num_trackers"]
        addr = :init_state => :init_trackers => i => :x
        init_obs[addr] = init_positions[i,1]
        addr = :init_state => :init_trackers => i => :y
        init_obs[addr] = init_positions[i,2]
    end
    
    # compiling further observations for the model
    args = [(t, params) for t in 1:T]
    observations = Vector{Gen.ChoiceMap}(undef, T)
    for t = 1:T
        cm = Gen.choicemap()
        cm[:states => t => :masks] = masks[t,:]
        observations[t] = cm
    end
    
    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_masks_static,
                                        (0, params),
                                        init_obs,
                                        args,
                                        observations)
    
    procedure = PopParticleFilter(params["inference_params"]["num_particles"],
                                  params["inference_params"]["num_particles"]/2, # ESS is in terms of effective particle count, not fraction
                                  nothing,
                                  tuple(),
                                  rejuvenate_state!, # rejuvenation
                                  retrieve_confusability, # population statistic
                                  early_stopping_confusability, # stopping criteria
                                  params["inference_params"]["max_rejuv"],
                                  params["inference_params"]["early_stopping_steps"], # early stopping criteria
                                  true)
    
    println("running inference...")
    results = sequential_monte_carlo(procedure, query,
                                     buffer_size = T,
                                     path = nothing)
    
    return results
end


