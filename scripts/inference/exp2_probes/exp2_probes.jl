using CSV
using MOT
using ArgParse
using Accessors
using Gen_Compose

# using Random
# Random.seed!(1234);
# @warn "Seed is set, remove before experiments!"

# using Profile
# using StatProfilerHTML

experiment_name = "exp2_probes"

exp_params = (;experiment_name = "exp2_probes",
              gm = "$(@__DIR__)/gm.json",
              proc = "$(@__DIR__)/proc.json",
              att = "$(@__DIR__)/ac.json",
              dataset = "/spaths/datasets/$(experiment_name).json",
              dur = 480,
              restart = true,
              viz = false,
              model = "adaptive_computation"
              )

function run_model(scene::Int, chain::Int)
    gm = dgp_gm = MOT.load(InertiaGM, exp_params.gm)
    # loading scene data
    scene_data = MOT.load_scene(dgp_gm,
                                exp_params.dataset,
                                scene)
    gt_states = scene_data[:gt_states][1:exp_params.dur]
    aux_data = scene_data[:aux_data]

    gm = setproperties(gm,
                       (n_dots = gm.n_targets + aux_data["n_distractors"],
                        vel = aux_data["vel"] * 0.55))

    query = query_from_params(gm, gt_states)

    att = MOT.load(PopSensitivity,
                   exp_params.att,
                   plan = td_flat,
                   plan_args = (1.025,),
                   percept_update = tracker_kernel,
                   percept_args = (3,) # look back steps
                   )
    proc = MOT.load(PopParticleFilter,
                    exp_params.proc;
                    attention = att)

    path = "/spaths/experiments/$(experiment_name)_$(exp_params.model)/$(scene)"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end

    c = args["chain"]
    logger = MemLogger(length(gt_states))
    chain_perf_path = joinpath(path, "$(c)_perf.csv")
    chain_att_path = joinpath(path, "$(c)_att.csv")

    println("running chain $c")

    if isfile(chain_perf_path) && args["restart"]
        rm(chain_perf_path)
        rm(chain_att_path)
    end

    chain = run_chain(proc, query, length(gt_states), logger)

    dg = extract_digest(chain)
    pf = MOT.chain_performance(chain, dg,
                               n_targets = gm.n_targets)
    pf[!, :scene] .= args["scene"]
    pf[!, :chain] .= c
    CSV.write(chain_perf_path, pf)
    af = MOT.chain_attention(chain, dg,
                             n_targets = gm.n_targets)
    af[!, :scene] .= args["scene"]
    af[!, :chain] .= c
    CSV.write(chain_att_path, af)

    args["viz"] && visualize_inference(chain, dg, gt_states, gm,
                                       joinpath(path, "$(c)_scene"))
    return nothing
end

function parse_args()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "scene"
        help = "Which scene to run"
        arg_type = Int64
        default = 24

        "chain"
        help = "chain id"
        arg_type = Int64
        default = 1
    end

    return parse_args(s)
end

function main()
    args = parse_args()
    i = args["scene"]
    c = args["chain"]
    run_model(i, c);
end


main();
