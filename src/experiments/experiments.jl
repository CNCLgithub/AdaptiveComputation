function run_inference(query::SequentialQuery,
                       proc::Gen_Compose.AbstractParticleFilter)

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = length(query))
end
function run_inference(query::SequentialQuery,
                       proc::Gen_Compose.AbstractParticleFilter,
                       path::String)

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = length(query),
                                     path = path)
end

function query_from_params(gm_params_path::T, dataset::T, scene::K, k::K;
                           gm = gm_brownian_mask, dm = nothing,
                           rf_params = nothing, fmasks_decay_function = nothing) where {T<:String, K<:Int64}

    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => isnothing(rf_params) ? extract_assignments : extract_assignments_receptive_fields,
               :causal_graph => extract_causal_graph,
               :trace => extract_trace)
               #:tracker_masks => extract_tracker_masks)

    latent_map = LatentMap(_lm)

    gm_params = load(GMParams, gm_params_path, fmasks_decay_function = fmasks_decay_function)

    scene_data = load_scene(scene, dataset, gm_params;
                            generate_masks=true,
                            k=k)

    @show scene_data[:aux_data]

    dm = isnothing(dm) ? scene_data[:dm] : dm
    masks = scene_data[:masks]
    gt_causal_graphs = scene_data[:gt_causal_graphs]

    # initial observations based on init_positions
    # model knows where trackers start off
    constraints = Gen.choicemap()
    init_dots = get_objects(gt_causal_graphs[1], Dot)

    for i=1:gm_params.n_trackers
        addr = :init_state => :trackers => i => :x
        constraints[addr] = init_dots[i].pos[1]
        addr = :init_state => :trackers => i => :y
        constraints[addr] = init_dots[i].pos[2]
    end
    
    receptive_fields = nothing

    # compiling further masks for the model
    if isnothing(rf_params)
        args = [(t, dm, gm_params) for t in 1:k]

        observations = Vector{Gen.ChoiceMap}(undef, k)
        for t = 1:k
            cm = Gen.choicemap()
            cm[:kernel => t => :masks] = masks[t]
            observations[t] = cm
        end
    else
        receptive_fields = get_rectangle_receptive_fields(rf_params.rf_dims..., gm_params, overlap = rf_params.overlap)
        args = [(t, dm, gm_params, receptive_fields, rf_params.rf_prob_threshold) for t in 1:k]

        observations = Vector{Gen.ChoiceMap}(undef, k)
        for t = 1:k
            cm = Gen.choicemap()
            
            cropped_masks = @>> receptive_fields map(rf -> cropfilter(rf, masks[t]))
            @show cropfilter(receptive_fields[5], [masks[t][3]])

            for i=1:length(receptive_fields)
                cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
            end
            observations[t] = cm
        end
    end
   

    if isnothing(rf_params)
        init_args = (0, dm, gm_params)
    else
        init_args = (0, dm, gm_params, receptive_fields, rf_params.rf_prob_threshold)
    end

    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm,
                                        init_args,
                                        constraints,
                                        args,
                                        observations)

    return query, gt_causal_graphs, gm_params, receptive_fields
end

export run_inference, query_from_params
