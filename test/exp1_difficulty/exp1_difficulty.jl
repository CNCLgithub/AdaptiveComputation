using MOT
using MOT: CausalGraph
using Gen_Compose
using Setfield

using Profile
using StatProfilerHTML

using Random
Random.seed!(1)

experiment_name = "exp1_difficulty_test"

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm" => "$(@__DIR__)/dm.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "dataset_path" => "/spaths/datasets/exp1_difficulty.json",
                 "k" => 240,
                 "scene" => 10,
                 "step_size" => 60,
                 "viz" => true,
                 "resume" => false])
    
    # loading scene data
    scene_data = MOT.load_scene(args["dataset_path"],
                                args["scene"])
    gt_cgs = scene_data[:gt_causal_graphs][1:args["k"]]
    aux_data = scene_data[:aux_data]

    gm = MOT.load(GMParams, args["gm"])
    gm = @set gm.n_targets = sum(aux_data["targets"]) # always 4 targets but whatever
    gm = @set gm.max_things = gm.n_targets + aux_data["n_distractors"]

    dm = MOT.load(InertiaModel, args["dm"])
    dm = @set dm.vel = aux_data["vel"]

    graphics = MOT.load(Graphics, args["graphics"])

    query = query_from_params(gt_cgs,
                              dgp,
                              gm_inertia_mask,
                              gm,
                              dm,
                              graphics,
                              length(gt_cgs))

    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"],
                   objective = MOT.td_flat,
                   )

    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    path = "/spaths/experiments/$(experiment_name)_$(att_mode)/$(args["scene"])"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end

    c = 1
    chain_path = joinpath(path, "$(c).jld2")
    # Profile.init(delay = 0.001,
    #              n = 10^6)
    # #@profilehtml results = run_inference(query, proc)
    # #@profilehtml results = run_inference(query, proc)
    isfile(chain_path) && !args["resume"] && rm(chain_path)
    if isfile(chain_path) && args["resume"]
        chain  = resume_chain(chain_path, args["step_size"])
    else
        chain = sequential_monte_carlo(proc, query, chain_path,
                                    args["step_size"])
        # chain = sequential_monte_carlo(proc, query, nothing,
        #                             args["step_size"])
    end

    df = MOT.chain_performance(chain, chain_path,
                               n_targets = gm.n_targets)
    display(df)

    if (args["viz"])
        visualize_inference(chain, chain_path, gt_cgs, gm,
                            graphics, att, path)
    end

    # return results
end

results = main();
