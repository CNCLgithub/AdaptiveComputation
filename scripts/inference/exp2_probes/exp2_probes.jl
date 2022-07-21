using CSV
using GenRFS
using MOT
using Gen_Compose
using ArgParse
using Setfield

# using Random
# Random.seed!(1235);

# using Profile
# using StatProfilerHTML

experiment_name = "exp2_probes"

function parse_commandline(vs)
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

        "--graphics"
        help = "Graphics params"
        arg_type = String
        default = "$(@__DIR__)/graphics.json"

        "--dm"
        help = "Motion parameters for Inertia model"
        arg_type = String
        default = "$(@__DIR__)/dm.json"

        "--dataset"
        help = "jld2 dataset path"
        arg_type = String
        default = "/spaths/datasets/$(experiment_name).json"

        "--time", "-t"
        help = "How many frames"
        arg_type = Int64
        default = 480

        "--step_size", "-s"
        help = "How many steps before saving"
        arg_type = Int64
        default = 30

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

        "--objective"
        help = "Attention objective"
        arg_type = Function
        default = td_flat
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

    return parse_args(vs, s)
end

function run(cmd)
    args = parse_commandline(cmd)
    display(args)

    # loading scene data
    scene_data = MOT.load_scene(args["dataset"],
                                args["scene"])
    gt_cgs = scene_data[:gt_causal_graphs][1:args["time"]]
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
    att = MOT.load(MapSensitivity,
                   args[att_mode]["params"],
                   objective = args[att_mode]["objective"],
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

    c = args["chain"]
    chain_path = joinpath(path, "$(c).jld2")

    println("running chain $c")

    isfile(chain_path) && args["restart"] && rm(chain_path)
    if isfile(chain_path)
        chain  = resume_chain(chain_path, args["step_size"])
    else
        chain = sequential_monte_carlo(proc, query, chain_path,
                                    args["step_size"])
    end
    # end

    pf = MOT.chain_performance(chain, chain_path,
                               n_targets = gm.n_targets)
    pf[!, :scene] .= args["scene"]
    pf[!, :chain] .= c
    CSV.write(joinpath(path, "$(c)_perf.csv"), pf)
    af = MOT.chain_attention(chain, chain_path,
                             n_targets = gm.n_targets)
    af[!, :scene] .= args["scene"]
    af[!, :chain] .= c
    CSV.write(joinpath(path, "$(c)_att.csv"), af)

    if (args["viz"])
        visualize_inference(chain, chain_path, gt_cgs, gm,
                            graphics, att, path)
    end

    return nothing
end



function main()
    # args = Dict("scene" => 1,
    #             "chain" => 1)
    args = parse_outer()
    i = args["scene"]
    c = args["chain"]
    # scene, chain, time

    cmd = ["$(i)", "$c", "T"]
    # cmd = ["$(i)", "$c", "-v", "-r", "--time=10", "T"]
    run(cmd);
end

function parse_outer()
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

main();
