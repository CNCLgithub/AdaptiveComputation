export Exp0

@with_kw struct Exp0 <: AbstractExperiment
    k::Int = 120
    trial::Int = 1
    gm::String = "$(@__DIR__)/gm.json"
    proc::String = "$(@__DIR__)/proc.json"
    dataset_path::String = "datasets/exp_0.h5"
end

get_name(::Exp0) = "exp0"

function run_inference(q::Exp0, attention::T, path::String; viz::Bool = false) where
    {T<:AbstractAttentionModel}

    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => extract_assignments)
    latent_map = LatentMap(_lm)

    gm_params = load(GMMaskParams, q.gm)

    # generating initial positions and masks (observations)
    init_positions, masks, motion, positions = load_exp0_trial(q.trial,
                                                               gm_params,
                                                               q.dataset_path)

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
        cm[:kernel => t => :masks] = masks[t]
        observations[t] = cm
    end

    query = Gen_Compose.SequentialQuery(latent_map,
                                        gm_brownian_mask,
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
        render(gm_params;
               dot_positions = positions[1:q.k],
               path = joinpath(out, "render"),
               pf_xy=tracker_positions[:,:,:,1:2],
               attended=attended/attention.sweeps,)

    end
    results
end


