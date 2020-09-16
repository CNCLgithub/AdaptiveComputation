export Exp1ISR

@with_kw struct Exp1ISR <: AbstractExperiment
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
    trial::Union{Nothing, Int} = nothing
    dataset_path::String = "/datasets/exp1_isr.jld2"
end

get_name(::Exp1ISR) = "exp1_isr"

function run_inference(q::Exp1ISR,
                       attention::T,
                       path::String;
                       viz::Bool=true) where {T<:AbstractAttentionModel}
    
    gm = load(GMMaskParams, q.gm)
    motion = load(InertiaModel, q.motion)

    trial_data = load_trial(q.trial, q.dataset_path, gm;
                            generate_masks=true)
    masks = trial_data[:masks]
    gt_causal_graphs = trial_data[:gt_causal_graphs]
    
    latent_map = LatentMap(Dict(
        :tracker_positions => extract_tracker_positions,
        :assignments => extract_assignments,
        :causal_graph => extract_causal_graph))


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
    args = [(t, motion, gm) for t in 1:q.k]
    observations = Vector{Gen.ChoiceMap}(undef, q.k)
    for t = 1:q.k
        cm = Gen.choicemap()
        cm[:kernel => t => :masks] = masks[t]
        observations[t] = cm
    end
    

    query = Gen_Compose.SequentialQuery(latent_map,
                                        #gm_isr_mask,
                                        gm_inertia_mask,
                                        (0, motion, gm),
                                        constraints,
                                        args,
                                        observations)

    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))
    
    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = path)
    
    if viz
        visualize_inference(results, gt_causal_graphs,
                            gm, attention, dirname(path))
    end

    return results
end

