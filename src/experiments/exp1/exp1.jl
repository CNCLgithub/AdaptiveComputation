export Exp1

@with_kw struct Exp1 <: AbstractExperiment
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
    scene::Int = 1
    dataset_path::String = "/datasets/exp1_brownian.jld2"
    fmasks::Bool = false
    fmasks_decay_function::Function = x->x
    fmasks_n::Int = 5
end

get_name(::Exp1) = "exp1"

function run_inference(q::Exp1,
                       attention::T,
                       path::String;
                       viz::Bool=true) where {T<:AbstractAttentionModel}
    
    if !q.fmasks
        gm = load(GMMaskParams, q.gm)
    else
        gm = load(GMMaskParams, q.gm;
                  fmasks=true,
                  fmasks_decay_function=q.fmasks_decay_function,
                  fmasks_n=q.fmasks_n)
    end

    display(gm)
    
    scene_data = load_scene(q.scene, q.dataset_path, gm;
                            generate_masks=true)
    masks = scene_data[:masks]
    gt_causal_graphs = scene_data[:gt_causal_graphs]
    motion = scene_data[:motion]
    latent_map = LatentMap(Dict(
                                :causal_graph => extract_causal_graph,
                                :assignments => extract_assignments
                               ))

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
   
    # TODO find a cleaner way of doing this (maybe we should have separate experiments?)
    if occursin("brownian", q.dataset_path)
        gm_function = gm_brownian_mask
    elseif occursin("isr", q.dataset_path)
        gm_function = gm_isr_mask
    else
        error("unrecognized dataset, not sure which generative function to load")
    end

    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_function,
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

