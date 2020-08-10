export ISRDynamicsExperiment

@with_kw struct ISRDynamicsExperiment <: AbstractExperiment
    proc::String = "$(@__DIR__)/proc.json"
    gm::String = "$(@__DIR__)/gm.json"
    motion::String = "$(@__DIR__)/motion.json"
    attention::String = "$(@__DIR__)/attention.json"
    k::Int = 120
end

get_name(::ISRDynamicsExperiment) = "example"

function run_inference(q::ISRDynamicsExperiment, path::String; viz::Bool=false)

    gm_params = load(GMMaskParams, q.gm)
    motion_data = ISRDynamics()
    
    # generating initial positions and masks (observations)
    init_positions, masks, positions = dgp(q.k, gm_params, motion_data)

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions
                               ))

    motion = load(ISRDynamics, q.motion)
    
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
                                        gm_masks_isr_static,
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
    if viz
        extracted = extract_chain(results)
        tracker_positions = extracted["unweighted"][:tracker_positions]
        # tracker_masks = get_masks(tracker_positions)
        aux_state = extracted["aux_state"]
        attention_weights = [aux_state[t].stats for t = 1:q.k]
        attention_weights = collect(hcat(attention_weights...)')

        out = dirname(path)
        plot_compute_weights(attention_weights, out)

        attempts = Vector{Int}(undef, q.k)
        attended = Vector{Vector{Float64}}(undef, q.k)
        for t=1:q.k
            attempts[t] = aux_state[t].attempts
            attended[t] = aux_state[t].attended_trackers
        end
        MOT.plot_attention(attended, attention.sweeps, out)
        plot_rejuvenation(attempts, out)

        # visualizing inference on stimuli
        render(positions, gm_params;
               path = joinpath(out, "render"),
               pf_xy=tracker_positions[:,:,:,1:2],
               attended=attended/attention.sweeps,)

    end

    #full_imgs = get_full_imgs(masks)
    #visualize(tracker_positions, full_imgs, gm_params)

    return results
end


