function run_inference(query::SequentialQuery,
                       proc::Gen_Compose.AbstractParticleFilter,
                       path::String;
                       viz::Bool = False) where
    {T<:AbstractAttentionModel}

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = length(query),
                                     path = path)
    if viz
        visualize_inference(results, query, dirname(path))
    end
    return results
end

function query_from_params(gm_params_path::T, dataset::T, scene::K, k::K;
                           gm = gm_brownian_mask, motion = nothing) where
    {T<:String, K<:Int}

    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => extract_assignments,
               :causal_graph => extract_causal_graph)
    latent_map = LatentMap(_lm)

    gm_params = load(GMMaskParams, gm_params_path)

    scene_data = load_scene(scene, dataset_path, gm_params;
                            generate_masks=true)

    motion = isnothing(motion) ? scene_data[:motion] : motion
    masks = scene_data[:masks]
    gt_causal_graphs = scene_data[:gt_causal_graphs]

    # initial observations based on init_positions
    # model knows where trackers start off
    constraints = Gen.choicemap()
    init_dots = gt_causal_graphs[1].elements
    for i=1:gm.n_trackers
        addr = :init_state => :trackers => i => :x
        constraints[addr] = init_dots[i].pos[1]
        addr = :init_state => :trackers => i => :y
        constraints[addr] = init_dots[i].pos[2]
    end

    # compiling further observations for the model
    args = [(t, motion, gm_params) for t in 1:k]
    observations = Vector{Gen.ChoiceMap}(undef, k)
    for t = 1:k
        cm = Gen.choicemap()
        cm[:kernel => t => :masks] = masks[t]
        observations[t] = cm
    end

    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm,
                                        (0, motion, gm_params),
                                        constraints,
                                        args,
                                        observations)
end

export run_inference, query_from_params
