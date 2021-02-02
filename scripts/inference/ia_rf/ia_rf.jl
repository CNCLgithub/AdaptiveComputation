"""
    inference script for ia_rf, i.e.
    individual_attention_receptive_fields
    (for the project with Qi)
"""


using CSV
using MOT
using Random
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
        help = "Motion parameters for dynamics model"
        arg_type = String
        default = "$(@__DIR__)/motion.json"

        "--attention"
        help = "Attention parameters (target designation using sensitivity)"
        arg_type = String
        default = "$(@__DIR__)/attention.json"

        "--rf_params"
        help = "Receptive fields parameters"
        arg_type = String
        default = "$(@__DIR__)/rf.json"

        "--dataset"
        help = "path to the dataset"
        arg_type = String
        default = joinpath("/datasets", "ia_mot.jld2")

        "--time", "-t"
        help = "How many frames"
        arg_type = Int64
        default = 299

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

        "compute"
        help = "The amount of compute (max sweeps for attention"
        arg_type = Int
        required = true

        "n_targets"
        help = "Number of targets"
        arg_type = Int
        required = true

    end

    return parse_args(s)
end

experiment_name = "individual_attention_receptive_fields"

function main()
    #args = parse_commandline()
    dir = "$(@__DIR__)"
    args = Dict("scene" => 1,
                "chain" => 1,
                "compute" => 20,
                "n_targets" => 3,
                "viz" => true,
                "gm" => "$(dir)/gm.json",
                "proc" => "$(dir)/proc.json",
                "motion" => "$(dir)/motion.json",
                "attention" => "$(dir)/attention.json",
                "rf_params" => "$(dir)/rf.json",
                "dataset" => "/datasets/ia_mot.jld2",
                "time" => 299,
                "restart" => true)

    Random.seed!(1)
    
    prob_threshold = 0.01
    n_targets = args["n_targets"]
    n_dots = 12.0
    n_distractors = n_dots - n_targets

    gm_params = MOT.load(GMMaskParams, args["gm"],
                         n_trackers = n_targets,
                         distractor_rate = n_distractors)
    display(gm_params)

    att = MOT.load(MapSensitivity, args["attention"],
                   objective = MOT.target_designation_receptive_fields,
                   sweeps = args["compute"])
    #att = UniformAttention(sweeps = 2)
    motion = MOT.load(InertiaModel, args["motion"])
    #motion = MOT.load(BrownianDynamicsModel, args["motion"])
    rf_params = MOT.load(RectRFParams, args["rf_params"])

    lm = Dict(:causal_graph => MOT.extract_causal_graph)
    lm_end = Dict(:assignments => MOT.extract_assignments_receptive_fields)

    # crop observations into receptive fields
    receptive_fields = get_rectangle_receptive_fields(rf_params.n_x, rf_params.n_y,
                                                      gm_params,
                                                      overlap = rf_params.overlap)

    query, gt_causal_graphs, masks = query_from_params(gm_params, args["dataset"],
                                                args["scene"], args["time"],
                                                #gm = gm_receptive_fields_brownian,
                                                gm = gm_receptive_fields,
                                                motion = motion,
                                                lm = lm,
                                                lm_end = lm_end,
                                                receptive_fields = receptive_fields,
                                                prob_threshold = prob_threshold)
    
    display(gt_causal_graphs)


    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    base_path = "/experiments/$(experiment_name)"
    scene = args["scene"]
    path = joinpath(base_path, "$(scene)")
    try
        isdir(base_path) || mkpath(base_path)
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    chain = args["chain"]
    compute = args["compute"]
    out = joinpath(path, "$(compute)_$(chain).jld2")
    if isfile(out) && !args["restart"]
        println("chain $chain for compute $compute complete")
        return
    end

    println("running chain $chain for compute $compute")
    results = run_inference(query, proc)

    if (args["viz"])
        visualize_inference(results, gt_causal_graphs,
                            gm_params, att, dirname(path),
                            receptive_fields = (rf_params.n_x, rf_params.n_y),
                            receptive_fields_overlap = rf_params.overlap)
    end
    
    df = MOT.analyze_chain_receptive_fields(results, n_targets,
                                            n_dots,
                                            receptive_fields,
                                            masks[args["time"]])
                           
    df[!, :scene] .= args["scene"]
    df[!, :chain] .= chain
    df[!, :compute] .= compute
    CSV.write(joinpath(path, "$(chain)_$(compute).csv"), df)

    return nothing
end


main();


