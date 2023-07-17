using CSV
using Gen
using MOT
using ArgParse
using Accessors
using Statistics
using Gen_Compose

################################################################################
# Global variables
################################################################################

experiment_name = "exp3_localization_error"
nobjects = 10
ntargets = 3
plan = :td

exp_params = (;
              gm = "$(@__DIR__)/gm.json",
              proc = "$(@__DIR__)/proc.json",
              att = "$(@__DIR__)/$(plan).json",
              dur = 120, # number of frames to run; full = 120
              model = "adaptive_computation",
              dataset = "/spaths/datasets/$(experiment_name).json",
              # SET FALSE for full experiment
              restart = false,
              viz = false,
              # restart = true,
              # viz = true,
              )

plan_objectives = Dict(
    # key => (plan object, args)
    :td => (td_flat, (1.025,)),
    :eu => (ensemble_uncertainty, (1.0, ))
)

plan_obj, plan_args = plan_objectives[plan]

################################################################################
# Main call
################################################################################

function run_model(scene::Int, chain::Int)
    gm = MOT.load(InertiaGM, exp_params.gm)

    # loading scene data
    scene_data = MOT.load_scene(gm,
                                exp_params.dataset,
                                scene)
    gt_states = scene_data[:gt_states][1:exp_params.dur]
    aux_data = scene_data[:aux_data]

    gm = setproperties(gm,
                       (n_dots = nobjects,
                        n_targets = ntargets,
                        target_p = ntargets / nobjects,
                        vel = aux_data["vel"] * 0.55))

    query = query_from_params(gm, gt_states)

    # attention module and particle filter
    att = MOT.load(PopSensitivity,
                   exp_params.att,
                   plan = plan_obj,
                   plan_args = plan_args,
                   percept_update = tracker_kernel,
                   percept_args = (3,), # look back steps
                   latents = ntargets
                   )
    proc = MOT.load(PopParticleFilter,
                    exp_params.proc;
                    attention = att)

    path = "/spaths/experiments/$(experiment_name)_$(exp_params.model)_$(plan)/$(scene)"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end

    # determine if inference should be reset
    # if previous checkpoint is found
    perf_path = joinpath(path, "$(chain)_perf.csv")
    att_path = joinpath(path, "$(chain)_att.csv")
    if isfile(perf_path) && exp_params.restart
        # restartng will remove previous
        # checkpoints
        rm(perf_path)
        rm(att_path)
    end

    # run the inference chain across all observations
    nsteps = length(query)
    logger = MemLogger(nsteps)
    println("running for $(nsteps) steps")
    smc_chain = run_chain(proc, query, nsteps, logger)

    # process results
    dg = extract_digest(logger)
    perf_df = MOT.chain_performance(dg)
    att_df = MOT.chain_attention(dg, ntargets)
    # append step data
    perf_df[!, :chain] .= chain
    perf_df[!, :scene] .= scene
    att_df[!, :chain] .= chain
    att_df[!, :scene] .= scene


    # store results
    CSV.write(perf_path, perf_df)
    CSV.write(att_path, att_df)
    exp_params.viz && visualize_inference(smc_chain, dg, gt_states, gm,
                                            joinpath(path, "$(chain)"))

    return nothing
end

function pargs()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "scene"
        help = "Scene / trial to run"
        arg_type = Int64
        default = 1

        "chain"
        help = "chain id"
        arg_type = Int64
        default = 1
    end

    return parse_args(s)
end

function main()
    args = pargs()
    i = args["scene"]
    c = args["chain"]
    run_model(i, c);
end


main();
