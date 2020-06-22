export Exp0Attention

@with_kw struct Exp0Attention <: AbstractExperiment
    trial::Int
    save_path::Union{Nothing,String} = nothing
    dataset_path::String = "datasets/exp_0.h5"
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
end

get_name(::Exp0Attention) = "exp0_attention"

function run_inference(q::Exp0Attention)

    gm_params = load(GMMaskParams, q.gm)
    
    # generating initial positions and masks (observations)
    init_positions, masks, motion = load_exp0_trial(q.trial, gm_params, q.dataset_path)

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                                :assignments => extract_assignments
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
        cm[:states => t => :masks] = masks[t]
        observations[t] = cm
    end
    
    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_masks_static,
                                        (0, motion, gm_params),
                                        constraints,
                                        args,
                                        observations)

    attention = load(TDEntropyAttentionModel, q.attention;
                     perturb_function = perturb_state!)

    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))
    

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = q.save_path)
    
    extracted = extract_chain(results)
    tracker_positions = extracted["unweighted"][:tracker_positions]

    final_assignments = extracted["weighted"][:assignments][q.k,:,:]
    final_log_scores = extracted["log_scores"][q.k,:]
    
    for i in sortperm(final_log_scores, rev=true)
        println("particle $i   A $(final_assignments[i,:])    log_score $(final_log_scores[i])")
    end

    return results
end

