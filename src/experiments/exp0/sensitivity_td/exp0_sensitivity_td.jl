export Exp0SensTD

@with_kw struct Exp0SensTD <: AbstractExperiment
    trial::Int = 1
    dataset_path::String = "datasets/exp_0.h5"
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
end

get_name(::Exp0SensTD) = "exp0_senstd"

function run_inference(q::Exp0SensTD, path::String)

    gm_params = load(GMMaskParams, q.gm)
    
    # generating initial positions and masks (observations)
    init_positions, masks, motion, positions = load_exp0_trial(q.trial, gm_params, q.dataset_path)

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                                :assignments => extract_assignments,
                                :tracker_masks => extract_tracker_masks
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

    attention = load(MapSensitivity, q.attention)

    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))
    

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = nothing)
    

    extracted = extract_chain(results)
    tracker_positions = extracted["unweighted"][:tracker_positions]
    tracker_masks = extracted["unweighted"][:tracker_masks]
    aux_state = extracted["aux_state"]
    attention_weights = [aux_state[t].stats for t = 1:q.k]
    attention_weights = collect(hcat(attention_weights...)')
    mkpath(path)
    plot_compute_weights(attention_weights, path)
    
    attempts = Vector{Int}(undef, q.k)
    attended = Vector{Vector{Float64}}(undef, q.k)
    for t=1:q.k
        attempts[t] = aux_state[t].attempts
        attended[t] = aux_state[t].attended_trackers
    end

    plot_rejuvenation(attempts, path)
    
    # visualizing inference on stimuli
    render(positions, q, gm_params;
           pf_xy=tracker_positions,
           attended = attended / attention.sweeps,
           tracker_masks=tracker_masks,
           dir = joinpath(path, "render"))

    plot_attention(attended, attention.sweeps, path)

    return results
end


