export ISRDynamicsExperiment

@with_kw struct ISRDynamicsExperiment <: AbstractExperiment
    trial::Union{Nothing, Int} = nothing
    dataset_path::String = "output/datasets/isr_dataset.jld2"
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
end

get_name(::ISRDynamicsExperiment) = "example"


function run_inference(q::ISRDynamicsExperiment,
                       attention::T,
                       path::String;
                      viz::Bool=true) where {T<:AbstractAttentionModel}
    
    gm = load(GMMaskParams, q.gm)
    motion = load(ISRDynamics, q.motion)
    att = MapSensitivity()
    
    if isnothing(q.trial)
        init_positions, init_vels, masks, positions = dgp(q.k, gm, motion)
    else
        init_positions, masks, motion, positions = load_trial(q.trial, q.dataset_path, gm)
    end

    # motion = BrownianDynamicsModel()
    # TODO change file path
    motion = load(ISRDynamics, "motion.json")

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                                :tracker_masks => extract_tracker_masks,
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
    args = [(t, motion, gm) for t in 1:q.k]
    observations = Vector{Gen.ChoiceMap}(undef, q.k)
    for t = 1:q.k
        cm = Gen.choicemap()
        cm[:kernel => t => :masks] = masks[t]
        observations[t] = cm
    end
    

    query = Gen_Compose.SequentialQuery(latent_map,
                                        #gm_isr_mask,
                                        gm_brownian_mask,
                                        (0, motion, gm),
                                        constraints,
                                        args,
                                        observations)

    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))
    

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = nothing)
    
    if viz
        visualize_inference(results, positions, gm, att, joinpath(path, "render"))
    end

    return results
end

