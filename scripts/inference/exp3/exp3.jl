using CSV
using MOT
using ArgParse
using Accessors
using Gen_Compose

experiment_name = "exp3"
plan = :td

exp_params = (;experiment_name = "exp2_probes",
              gm = "$(@__DIR__)/gm.json",
              proc = "$(@__DIR__)/proc.json",
              att = "$(@__DIR__)/$(plan).json",
              dataset = "/spaths/datasets/$(experiment_name).json",
              dur = 240, # number of frames to run; full = 240
              model = "adaptive_computation",
              # SET FALSE for full experiment
              restart = false,
              viz = false,
              )

plan_objectives = Dict(
    # key => (plan object, args)
    :td => (td_flat, (1.025,)),
    :eu => (ensemble_uncertainty, (1.0, ))
)

function run_model(scene::Int, chain::Int)
    gm = dgp_gm = MOT.load(InertiaGM, exp_params.gm)
    # loading scene data
    scene_data = MOT.load_scene(dgp_gm,
                                exp_params.dataset,
                                scene)
    gt_states = scene_data[:gt_states][1:exp_params.dur]
    aux_data = scene_data[:aux_data]

    n_targets = sum(aux_data["targets"])
    n_dots = length(aux_data["targets"])
    gm = setproperties(gm,
                       (n_dots = n_dots,
                        n_targets = n_targets,
                        target_p = n_targets / n_dots,
                        vel = aux_data["vel"] * 0.55))

    query = query_from_params(gm, gt_states)

    plan_obj, plan_args = plan_objectives[plan]
    att = MOT.load(PopSensitivity,
                   exp_params.att,
                   plan = plan_obj,
                   plan_args = plan_args,
                   percept_update = tracker_kernel,
                   percept_args = (3,), # look back steps
                   latents = n_targets
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

    nsteps = length(gt_states) - 1
    logger = MemLogger(nsteps)
    chain_perf_path = joinpath(path, "$(chain)_perf.csv")
    chain_att_path = joinpath(path, "$(chain)_att.csv")

    println("running chain $(chain)")
    if isfile(chain_perf_path) && exp_params.restart
        rm(chain_perf_path)
        rm(chain_att_path)
    end
    smc_chain = run_chain(proc, query, nsteps, logger)

    dg = extract_digest(logger)
    perf_df = MOT.chain_performance(dg)
    perf_df[!, :scene] .= scene
    perf_df[!, :chain] .= chain
    CSV.write(chain_perf_path, perf_df)
    att_df = MOT.chain_attention(dg, gm.n_targets)
    att_df[!, :scene] .= scene
    att_df[!, :chain] .= chain
    CSV.write(chain_att_path, att_df)

    exp_params.viz && visualize_inference(smc_chain, dg, gt_states, gm,
                                          joinpath(path, "$(chain)_scene"))
    return nothing
end

function pargs()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "scene"
        help = "Which scene to run"
        arg_type = Int64
        default = 36

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
