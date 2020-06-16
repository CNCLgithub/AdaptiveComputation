export run_inference


function run_inference(masks, init_positions, params)

    T = size(masks, 1)

    println("preparing for inference...")

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                               ))

    
    # initial observations based on init_positions
    # model knows where trackers start off
    constraints = Gen.choicemap()
    for i=1:params["query_params"].num_trackers
        addr = :init_state => :init_trackers => i => :x
        constraints[addr] = init_positions[i,1]
        addr = :init_state => :init_trackers => i => :y
        constraints[addr] = init_positions[i,2]
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
                                        constraints,
                                        args,
                                        observations)
    
    procedure = PopParticleFilter(params["inference_params"])

    println("running inference...")
    results = sequential_monte_carlo(procedure, query,
                                     buffer_size = T,
                                     path = nothing)
    
    return results
end


