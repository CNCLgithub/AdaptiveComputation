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


function get_observations(graphics::AbstractGraphics, masks)
    k = length(cgs)
    observations = Vector{Gen.ChoiceMap}(undef, k)
    receptive_fields = graphics.receptive_fields
    
    for t=1:k
        cm = Gen.choicemap()

        if receptive_fields isa NullReceptiveFields
            cm[:kernel => t => :masks] = masks[t]
        else
            cropped_masks = @>> receptive_fields map(rf -> cropfilter(rf, masks[t]))
            for i=1:length(receptive_fields)
                cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
            end
            observations[t] = cm
        end
    end
    
    return observations
end


function get_init_constraints(cg::CausalGraph)
    cm = Gen.choicemap()

    init_dots = get_objects(gt_causal_graphs[1], Dot)
    for i=1:gm_params.n_trackers
        addr = :init_state => :trackers => i => :x
        cm[addr] = init_dots[i].pos[1]
        addr = :init_state => :trackers => i => :y
        cm[addr] = init_dots[i].pos[2]
    end

    return cm
end


function query_from_params(gm_func::Function,
                           gm_params::AbstractGMParams,
                           dm_params::AbstractDynamicsModel,
                           graphics_params::AbstractGraphics,
                           k::Int64,
                           scene_data)

    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => isnothing(rf_params) ? extract_assignments : extract_assignments_receptive_fields,
               :causal_graph => extract_causal_graph,
               :trace => extract_trace)
               #:tracker_masks => extract_tracker_masks)
    latent_map = LatentMap(_lm)

    scene_data = load_scene(scene, dataset, gm_params;
                            generate_masks=true,
                            k=k)

    @show scene_data[:aux_data]

    masks = scene_data[:masks]
    gt_causal_graphs = scene_data[:gt_causal_graphs]
    init_cg = gt_causal_graphs[1]
    cgs = gt_causal_graphs[2:end]

    init_constraints = get_init_constraints(init_cg)
    observations = get_observations(graphics_params, masks)
  
    init_args = (0, cg)
    args = [(t,) for t in 1:k]

    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm,
                                        init_args,
                                        constraints,
                                        args,
                                        observations)

    return query, gt_causal_graphs, gm_params, receptive_fields
end

export run_inference, query_from_params
