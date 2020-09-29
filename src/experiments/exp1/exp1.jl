export Exp1

@with_kw struct Exp1 <: AbstractExperiment
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
    scene::Int = 1
    dataset_path::String = "/datasets/exp1.jld2"
end

get_name(::Exp1) = "exp1"

function run_inference(q::Exp1,
                       attention::T,
                       path::String;
                       viz::Bool=true) where {T<:AbstractAttentionModel}
    
    gm = load(GMMaskParams, q.gm)
    motion = load(BrownianDynamicsModel, q.motion)
    
    scene_data = load_scene(q.scene, q.dataset_path, gm;
                            generate_masks=true)
    masks = scene_data[:masks]
    gt_causal_graphs = scene_data[:gt_causal_graphs]
    motion = scene_data[:motion]


    latent_map = LatentMap(Dict(
                                :causal_graph => extract_causal_graph,
                                # :tracker_masks => extract_tracker_masks,
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
    gm_function = q.dataset_path == "/datasets/exp1.jld2" ? gm_brownian_mask : gm_isr_mask
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
                                     #path = joinpath(path, "results.jld2"))
                                     path = path)
    
    if viz
        visualize_inference(results, gt_causal_graphs,
                            gm, attention, dirname(path))
    end

    return results
end

