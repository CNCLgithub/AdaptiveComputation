function get_observations(graphics::Graphics, masks)
    k = size(masks, 1)
    observations = Vector{Gen.ChoiceMap}(undef, k)
    for t=1:k
        cm = Gen.choicemap()
        cm[:kernel => t => :masks] = masks[t]
        observations[t] = cm
    end
    return observations
end

function get_init_constraints(cg::CausalGraph)
    init_dots = get_objects(cg, Dot)
    get_init_constraints(cg, length(init_dots))
end

function get_init_constraints(cg::CausalGraph, n::Int64)
    #TODO: convert datasets into type agnostic format
    init_dots = only_targets(cg)
    n = length(init_dots)
    cm = Gen.choicemap()
    cm[:init_state => :n_trackers] = n
    for i=1:n
        dot = get_prop(cg, init_dots[i], :object)
        addr = :init_state => :trackers => i => :x
        cm[addr] = dot.pos[1]
        addr = :init_state => :trackers => i => :y
        cm[addr] = dot.pos[2]

        # TODO: add vel to `cg_from_positions`
        # vel = dot.vel
        # normv = norm(vel)
        # ang = vel ./ normv
        # ang = normv == 0. ? 0. : atan(ang[2], ang[1])
        # cm[:init_state => :trackers => i => :ang] = ang

        # by convention, the first n trackers are targets
        # in the source trace
        cm[:init_state => :trackers => i => :target] = true
    end
    return cm
end


function query_from_params(gt_causal_graphs,
                           dgp_params,
                           generative_model,
                           gm_params::GMParams,
                           dm_params::AbstractDynamicsModel,
                           graphics_params::Graphics,
                           k::Int64;
                           vis::Bool = false)
    
    latent_map = LatentMap(
        :auxillary => digest_auxillary,
        :parents => extract_parents,
        :positions => digest_tracker_positions
    )

    init_gt_cg = gt_causal_graphs[1]
    gt_cgs = gt_causal_graphs[2:end]

    init_constraints = get_init_constraints(init_gt_cg,
                                            gm_params.n_trackers)

    display(init_constraints)
    # ensure that all obs are present
    gr = @set graphics_params.bern_existence_prob = 1.0
    gr = @set gr.outer_f = 1.0
    gr = @set gr.inner_f = 1.0
    masks = render_from_cgs(gr,
                            gm_params,
                            gt_causal_graphs)
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
    ms = q.observations[:kernel => 1 => :masks]
    @debug "number of masks $(length(ms))"
    
    return query
end

export query_from_params
