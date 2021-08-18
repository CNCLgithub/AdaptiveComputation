using CSV
using MOT
using MOT: CausalGraph
using LightGraphs: SimpleDiGraph, add_vertex!
using MetaGraphs: set_prop!, get_prop
using ArgParse
using Setfield
using Profile
using StatProfilerHTML
using Lazy: @>>
using Statistics: norm, mean, std, var

experiment_name = "correlated_sensitivity"

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm" => "$(@__DIR__)/dm.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "dataset" => "/datasets/exp1_difficulty.jld2",
                 "scene" => 10,
                 "chain" => 1,
                 "time" => 100,
                 "restart" => true,
                 "viz" => true])

    # loading scene data
    scene_data = MOT.load_scene(args["scene"], args["dataset"])
    gt_cgs = scene_data[:gt_causal_graphs][1:args["time"]]
    aux_data = scene_data[:aux_data]

    gm = MOT.load(GMParams, args["gm"])
    gm = @set gm.n_trackers = sum(aux_data.targets) # always 4 targets but whatever
    gm = @set gm.distractor_rate = sum(aux_data.n_distractors)

    dm = MOT.load(InertiaModel, args["dm"])
    dm = @set dm.vel = aux_data[:vel]

    @show aux_data
    @show dm
    @show gm

    graphics = MOT.load(Graphics, args["graphics"])

    query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm,
                              dm,
                              graphics,
                              length(gt_cgs))

    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"],
                   objective = MOT.target_designation_receptive_fields,
                   # objective = MOT.data_correspondence_receptive_fields,
                   )
                   # weights = fill(-50.0, sum(scene_data[:targets])))

    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    path = "/experiments/$(experiment_name)_$(att_mode)"
    try
        isdir(base_path) || mkpath(base_path)
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    c = args["chain"]
    out = joinpath(path, "$(c).jld2")
    if isfile(out) && !args["restart"]
        println("chain $c complete")
        return
    end

    println("running chain $c")
    Profile.init(delay = 0.001,
                 n = 10^6)
    # @profilehtml results = run_inference(prof_query, proc)
    # @profilehtml results = run_inference(query, proc)
    results = run_inference(query, proc)

    df = MOT.analyze_chain_receptive_fields(results,
                                            n_trackers = gm_params.n_trackers,
                                            n_dots = gm_params.n_trackers + gm_params.distractor_rate,
                                            gt_cg_end = gt_cgs[end])

    if (args["viz"])
        visualize_inference(results, gt_cgs, gm_params,
                            graphics_params, att, path)
    end

    return results
end

results = main();
