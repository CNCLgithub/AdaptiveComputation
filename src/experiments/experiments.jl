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

function query_from_params(gm_params, dataset::T, scene::K, k::K;
                           gm = gm_brownian_mask, motion = nothing,
                           point_observations = false,
                           receptive_fields = nothing,
                           prob_threshold = 0.0001,
                           lm::Dict = Dict(),
                           lm_end::Dict = Dict()) where {T<:String, K<:Int}

    latent_map = LatentMap(lm)
    latent_map_end = LatentMap(lm_end)

    #gm_params = load(GMMaskParams, gm_params_path)

    scene_data = load_scene(scene, dataset, gm_params;
                            generate_masks=!point_observations)

    motion = isnothing(motion) ? scene_data[:motion] : motion
    masks = scene_data[:masks]
    gt_causal_graphs = scene_data[:gt_causal_graphs]

    # initial observations based on init_positions
    # model knows where trackers start off
    constraints = Gen.choicemap()
    init_dots = gt_causal_graphs[1].elements

    for i=1:gm_params.n_trackers
        addr = :init_state => :trackers => i => :x
        constraints[addr] = init_dots[i].pos[1]
        addr = :init_state => :trackers => i => :y
        constraints[addr] = init_dots[i].pos[2]
    end
    
    if isnothing(receptive_fields)
        # compiling further observations for the model
        args = [(t, motion, gm_params) for t in 1:k]
        observations = Vector{Gen.ChoiceMap}(undef, k)
        for t = 1:k
            cm = Gen.choicemap()
            cm[:kernel => t => :masks] = masks[t]
            observations[t] = cm
        end
        extra_gm_args = ()
    else
        args = [(t, motion, gm_params, receptive_fields) for t in 1:k]
        observations = Vector{Gen.ChoiceMap}(undef, k)
        for t = 1:k
            cm = Gen.choicemap()

            if point_observations
                points = @>> scene_data[:gt_causal_graphs][t].elements map(x -> x.pos)
                points_rf = @>> receptive_fields map(rf -> filter(p -> MOT.within(p, rf), points))
                display(points_rf)
                for i=1:length(receptive_fields)
                    cm[:kernel => t => :receptive_fields => i => :points] = points_rf[i]
                end
            else
                cropped_masks = @>> receptive_fields map(rf -> MOT.cropfilter(rf, masks[t]))
                for i=1:length(receptive_fields)
                    cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
                end
            end
            observations[t] = cm
        end
        extra_gm_args = (receptive_fields,)
    end

    query = Gen_Compose.SequentialQuery(latent_map,
                                        latent_map_end,
                                        gm,
                                        (0, motion, gm_params, extra_gm_args...),
                                        constraints,
                                        args,
                                        observations)

    return query, gt_causal_graphs, masks, scene_data
end

export run_inference, query_from_params
