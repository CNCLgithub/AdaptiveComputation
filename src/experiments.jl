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


function get_observations(graphics::Graphics, masks)
    k = size(masks, 1)
    observations = Vector{Gen.ChoiceMap}(undef, k)
    @unpack receptive_fields = graphics
    
    for t=1:k
        cm = Gen.choicemap()
        for i=1:length(receptive_fields)
            @debug "# of masks for rf $(i): $(length(masks[t][i]))"
            cm[:kernel => t => :receptive_fields => i => :masks] = masks[t][i]
        end
        observations[t] = cm
    end
    return observations
end

function constraints_from_cgs(cgs::Vector{CausalGraph},
                              gm::Gen.GenerativeFunction,
                              args::Tuple)
    t = length(cgs)
    # first simulate trace using gm
    cm = get_init_constraints(cgs[1])
    prev_objects = get_objects(cgs[1], Dot)
    for i = 2:t
        objects = MOT.get_objects(cgs[i], Dot)
        for j = 1:length(objects)
            pos = objects[j].pos[1:2]
            delta = pos - prev_objects[j].pos[1:2]
            nd = norm(delta)
            ang = delta ./ nd
            ang = nd == 0. ? 0. : atan(ang[2], ang[1])
            cm[:kernel => i-1 => :dynamics => :brownian => j => :mag] = nd
            cm[:kernel => i-1 => :dynamics => :brownian => j => :ang] = ang
        end
        prev_objects = objects
    end

    trace, _ = generate(gm, args, cm)
    choices = get_choices(trace)

    constraints = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        observations = choicemap()
        addr = :kernel => i => :receptive_fields
        set_submap!(observations, addr, get_submap(choices, addr))
        constraints[i] = observations
    end
    return constraints
end

function get_init_constraints(cg::CausalGraph)
    init_dots = get_objects(cg, Dot)
    get_init_constraints(cg, length(init_dots))
end
function get_init_constraints(cg::CausalGraph, n::Int64)
    cm = Gen.choicemap()
    init_dots = get_objects(cg, Dot)
    for i=1:n
        addr = :init_state => :trackers => i => :x
        cm[addr] = init_dots[i].pos[1]
        addr = :init_state => :trackers => i => :y
        cm[addr] = init_dots[i].pos[2]
    end
    return cm
end


function query_from_params(gt_causal_graphs,
                           dgp_params,
                           generative_model,
                           gm_params::AbstractGMParams,
                           dm_params::AbstractDynamicsModel,
                           graphics_params::Graphics,
                           k::Int64)
    
    if graphics_params.receptive_fields isa NullReceptiveFields
        assignments_func = extract_assignments
    else
        assignments_func = extract_assignments_receptive_fields
    end

    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => assignments_func,
               :causal_graph => extract_causal_graph,
               # :trace => extract_trace
               )
               #:tracker_masks => extract_tracker_masks)
    latent_map = LatentMap(_lm)

    init_gt_cg = gt_causal_graphs[1]
    gt_cgs = gt_causal_graphs[2:end]

    init_constraints = get_init_constraints(init_gt_cg,
                                            gm_params.n_trackers)

    display(init_constraints)

    masks = get_bit_masks_rf(gt_causal_graphs,
                             graphics_params,
                             gm_params)
    observations = get_observations(graphics_params, masks)

    init_args = (0, gm_params, dm_params, graphics_params)
    args = [(t, gm_params, dm_params, graphics_params) for t in 1:k]

    query = Gen_Compose.SequentialQuery(latent_map,
                                        generative_model,
                                        init_args,
                                        init_constraints,
                                        args,
                                        observations)




    q = first(query)
    ms = q.observations[:kernel => 1 => :receptive_fields => 1 => :masks]
    @debug "number of masks $(length(ms))"
    
    return query
end

export run_inference, query_from_params
