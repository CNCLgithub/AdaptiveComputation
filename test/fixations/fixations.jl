using CSV
using MOT
using ArgParse
using Setfield
using Profile
using StatProfilerHTML

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gm"
        help = "Generative Model params"
        arg_type = String
        default = "$(@__DIR__)/gm.json"

        "--proc"
        help = "Inference procedure params"
        arg_type = String
        default = "$(@__DIR__)/proc.json"

        "--dm"
        help = "Motion parameters for Inertia model"
        arg_type = String
        default = "$(@__DIR__)/dm.json"

        "--graphics"
        help = "Receptive fields parameters"
        arg_type = String
        default = "$(@__DIR__)/graphics.json"

        "--dataset"
        help = "jld2 dataset path"
        arg_type = String
        default = joinpath("/datasets", "fixations_dataset.jld2")

        "--fps", "-f"
        help = "Frames per step"
        arg_type = Int64
        default = 24

        "--time", "-t"
        help = "Number of steps"
        arg_type = Int64
        default = 600

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "--viz", "-v"
        help = "Whether to render masks"
        action = :store_true

        "scene"
        help = "Which scene to run"
        arg_type = Int
        required = true

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        required = true

        "target_designation", "T"
        help = "Using target designation"
        action = :command

        "data_correspondence", "D"
        help = "Using data correspondence"
        action = :command

        "scene_avg", "A"
        help = "Using scene avg"
        action = :command

    end

    @add_arg_table! s["target_designation"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/td.json"

    end
    @add_arg_table! s["data_correspondence"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/dc.json"
    end
    @add_arg_table! s["scene_avg"] begin
        "model_path"
        help = "path containing compute allocations"
        arg_type = String
        required = true

    end

    return parse_args(s)
end

experiment_name = "fixations"

function main()
    #args = parse_commandline()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm" => "$(@__DIR__)/dm.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "dataset" => "/datasets/fixations_dataset.jld2",
                 "scene" => 10,
                 "chain" => 1,
                 "fps" => 24,
                 "time" => 30,
                 "restart" => true,
                 "viz" => true])

    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"],
                   objective = MOT.target_designation_receptive_fields)
    
    scene_data = load_scene(args["scene"], args["dataset"])
    fps = round(Int64, 60 / args["fps"])
    t = args["time"]
    gt_cgs = scene_data[:gt_causal_graphs][1:fps:t]
    aux_data = scene_data[:aux_data]


    gm_params = MOT.load(GMParams, args["gm"])
    gm_params = @set gm_params.n_trackers = sum(aux_data[:targets])
    gm_params = @set gm_params.distractor_rate = count(iszero, aux_data[:targets])

    dm_params = MOT.load(InertiaModel, args["dm"])
    dm_params = @set dm_params.vel = aux_data[:vel_avg]

    graphics_params = MOT.load(Graphics, args["graphics"])


    prof_query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm_params,
                              dm_params,
                              graphics_params,
                              1)
    query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm_params,
                              dm_params,
                              graphics_params,
                              length(gt_cgs))

    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    base_path = "/experiments/$(experiment_name)_$(att_mode)"
    scene = args["scene"]
    path = joinpath(base_path, "$(scene)")
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
                                            gt_cg_end = gt_cgs[args["time"]])
    df[!, :scene] .= args["scene"]
    df[!, :chain] .= c
    CSV.write(joinpath(path, "$(c).csv"), df)

    if (args["viz"])
        visualize_inference(results, gt_cgs, gm_params,
                            graphics_params, att, path)
    end

    return results
end

results = main();
