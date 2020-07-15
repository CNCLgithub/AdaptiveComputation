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

function run_inference(q::Exp0SensTD, path::String; masks::Bool = false)
    attention = load(MapSensitivity, q.attention)
    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => extract_assignments)
    if masks
        _lm[:tracker_masks] = extract_tracker_masks
    end
    latent_map = LatentMap(_lm)
    results = run_inference(q, path,latent_map, attention)
    if masks
        extracted = extract_chain(results)
        tracker_positions = extracted["unweighted"][:tracker_positions]
        tracker_masks = extracted["unweighted"][:tracker_masks]
        aux_state = extracted["aux_state"]
        k = size(tracker_positions, 1)
        attention_weights = [aux_state[t].stats for t = 1:k]
        attention_weights = collect(hcat(attention_weights...)')

        out = dirname(path)

        plot_compute_weights(attention_weights, out)

        attempts = Vector{Int}(undef, k)
        attended = Vector{Vector{Float64}}(undef, k)
        for t=1:k
            attempts[t] = aux_state[t].attempts
            attended[t] = aux_state[t].attended_trackers
        end
        MOT.plot_attention(attended, 15, out)
        plot_rejuvenation(attempts, out)


        aux_state = extracted["aux_state"]
        attempts = Vector{Int}(undef, q.k)
        attended = Vector{Vector{Float64}}(undef, q.k)
        for t=1:q.k
            attempts[t] = aux_state[t].attempts
            attended[t] = aux_state[t].attended_trackers
        end
        # visualizing inference on stimuli
        gm_params = load(GMMaskParams, q.gm)
        positions = last(load_exp0_trial(q.trial, gm_params, q.dataset_path))
        render(positions, q, gm_params;
               dir = joinpath(out, "render"),
               pf_xy=tracker_positions[:,:,:,1:2],
               attended=attended/attention.sweeps,
               tracker_masks=tracker_masks)

    end
    results
end


function run_inference(q::Exp0SensTD, path::String, latent_map::LatentMap,
                       attention::MapSensitivity)
    gm_params = load(GMMaskParams, q.gm)

    # generating initial positions and masks (observations)
    init_positions, masks, motion, positions = load_exp0_trial(q.trial, gm_params, q.dataset_path)

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


    proc = load(PopParticleFilter, q.proc;
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))

    results = sequential_monte_carlo(proc, query,
                                     buffer_size = q.k,
                                     path = path)
    return results
end
