export Exp0

@with_kw struct Exp0 <: AbstractExperiment
    k::Int = 120
    trial::Int = 1
    gm::String = "$(@__DIR__)/gm.json"
    proc::String = "$(@__DIR__)/proc.json"
    dataset_path::String = "/datasets/exp0.jld2"
    #dataset_path::String = "/datasets/exp_0.h5"
end

get_name(::Exp0) = "exp0"

function run_inference(q::Exp0, attention::T, path::String; viz::Bool = false) where
    {T<:AbstractAttentionModel}

    _lm = Dict(:tracker_positions => extract_tracker_positions,
               :assignments => extract_assignments,
               :causal_graph => extract_causal_graph)
    latent_map = LatentMap(_lm)

    gm = load(GMMaskParams, q.gm)

    # getting some trial data (initial positions, ground truth causal graphs,
    # masks for observations, motion model/parameters) from the dataset
    # trial_data = load_exp0_trial(q.trial, gm, q.dataset_path;
                                 # generate_masks=true)
    trial_data = load_trial(q.trial, q.dataset_path, gm;
                            generate_masks=true)
    motion = trial_data[:motion]
    masks = trial_data[:masks]
    gt_causal_graphs = trial_data[:gt_causal_graphs]
    # init_positions = trial_data[:init_positions]

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
                                     path = path)

    if viz
        visualize_inference(results, gt_causal_graphs,
                            gm, attention, dirname(path))
    end
    results
end


