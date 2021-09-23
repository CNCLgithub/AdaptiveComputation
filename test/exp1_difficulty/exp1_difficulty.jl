using MOT
using MOT: CausalGraph
using Setfield
using Lazy: @>, @>>

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
                 "dataset_path" => "/datasets/exp1_difficulty.jld2",
                 "k" => 20,
                 "scene" => 1,
                 "viz" => true])
    
    # loading scene data
    scene_data = MOT.load_scene(args["scene"], args["dataset_path"])
    gt_cgs = scene_data[:gt_causal_graphs][1:args["k"]]
    aux_data = scene_data[:aux_data]

    gm = MOT.load(GMParams, args["gm"])
    @set gm.n_targets = sum(aux_data.targets) # always 4 targets but whatever
    @set gm.max_things = sum(aux_data.targets) + sum(aux_data.n_distractors)

    dm = MOT.load(InertiaModel, args["dm"])
    @set dm.vel = aux_data[:vel]

    graphics = MOT.load(Graphics, args["graphics"])

    query = query_from_params(gt_cgs,
                              dgp,
                              gm_inertia_mask,
                              gm,
                              dm,
                              graphics,
                              length(gt_cgs))

    # att_mode = "target_designation"
    # att = MOT.load(MapSensitivity, args[att_mode]["params"],
    #                objective = MOT.target_designation_receptive_fields,
    #                )

    # proc = MOT.load(PopParticleFilter, args["proc"];
    #                 rejuvenation = rejuvenate_attention!,
    #                 rejuv_args = (att,))

    # path = "/experiments/$(experiment_name)"
    # try
    #     isdir(path) || mkpath(path)
    # catch e
    #     println("could not make dir $(path)")
    # end

    # Profile.init(delay = 0.001,
    #              n = 10^6)
    # #@profilehtml results = run_inference(query, proc)
    # #@profilehtml results = run_inference(query, proc)
    # results = run_inference(query, proc)

    # df = MOT.analyze_chain_receptive_fields(results,
    #                                         n_trackers = gm.n_trackers,
    #                                         n_dots = gm.n_trackers + gm.distractor_rate,
    #                                         gt_cg_end = gt_cgs[end])

    # if (args["viz"])
    #     visualize_inference(results, gt_cgs, gm,
    #                         graphics, att, path)
    # end

    # return results
end

results = main();
