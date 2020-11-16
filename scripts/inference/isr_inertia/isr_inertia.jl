using CSV
using MOT
using ArgParse

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

        "--motion"
        help = "Motion parameters for Inertia model"
        arg_type = String
        default = "$(@__DIR__)/motion.json"

        "--dataset"
        help = "Motion parameters for Inertia model"
        arg_type = String
        default = joinpath("/datasets", "exp1_isr_480.jld2")

        "--time", "-t"
        help = "How many frames"
        arg_type = Int64
        default = 480

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

experiment_name = "isr_inertia_480"

function main()
    args = parse_commandline()
    att_mode = args["%COMMAND%"]
    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"])

    motion = MOT.load(InertiaModel, args["motion"])

    query, gt_causal_graphs, gm_params = query_from_params(args["gm"], args["dataset"],
                                                           args["scene"], args["time"],
                                                           gm = gm_inertia_mask,
                                                           motion = motion)

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
    results = run_inference(query, proc, out)

    if (args["viz"])
        visualize_inference(results, gt_causal_graphs, gm_params, att, path;
                            render_tracker_masks=true)
    end

    df = MOT.analyze_chain(results)
    df[!, :scene] .= args["scene"]
    df[!, :chain] .= c
    CSV.write(joinpath(path, "$(c).csv"), df)

    return nothing
end


main();
