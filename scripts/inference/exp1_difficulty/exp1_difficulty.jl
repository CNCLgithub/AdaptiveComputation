using CSV
using MOT
using Gen_Compose
using ArgParse
using Accessors

# using Random
# Random.seed!(1234);

# using Profile
# using StatProfilerHTML

experiment_name = "exp1_difficulty"
att_mode = "target_designation"
att_params = "$(@__DIR__)/td.json"
objective = td_flat

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

        "--dataset"
        help = "jld2 dataset path"
        arg_type = String
        default = "/spaths/datasets/$(experiment_name).json"

        "--time", "-t"
        help = "How many frames"
        arg_type = Int64
        default = 240

        "--step_size", "-s"
        help = "How many steps before saving"
        arg_type = Int64
        default = 60

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "--viz", "-v"
        help = "Whether to render masks"
        action = :store_true

        "--scene"
        help = "Which scene to run"
        arg_type = Int
        default = 50

        "--chain"
        help = "The number of chains to run"
        arg_type = Int
        default = 1
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    # args = default_args()


    # loading scene data
    gm = MOT.load(InertiaGM, args["gm"])
    scene_data = MOT.load_scene(gm,
                                args["dataset"],
                                args["scene"])
    gt_states = scene_data[:gt_states][1:args["time"]]
    aux_data = scene_data[:aux_data]
    gm = @set gm.n_dots = gm.n_targets + aux_data["n_distractors"]
    gm = @set gm.vel = aux_data["vel"] * 0.45
    gm = @set gm.w = gm.w * gm.vel

    query = query_from_params(gm, gt_states, length(gt_states))

    att = MOT.load(PopSensitivity,
                   att_params,
                   plan = td_flat,
                   plan_args = (),
                   percept_update = tracker_kernel,
                   percept_args = (4,) # look back steps
                   )

    proc = MOT.load(PopParticleFilter,
                    args["proc"];
                    attention = att)

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

    dg = extract_digest(chain_path)
    pf = MOT.chain_performance(chain, dg,
                               n_targets = gm.n_targets)
    pf[!, :scene] .= args["scene"]
    pf[!, :chain] .= c
    CSV.write(joinpath(path, "$(c)_perf.csv"), pf)
    af = MOT.chain_attention(chain, dg,
                             n_targets = gm.n_targets)
    af[!, :scene] .= args["scene"]
    af[!, :chain] .= c
    CSV.write(joinpath(path, "$(c)_att.csv"), af)

    render_pf(chain, joinpath(path, "$(c)_graphics"))
    visualize_inference(chain, dg, gt_states, gm,
                                       joinpath(path, "$(c)_scene"))
    # args["viz"] && render_pf(chain, joinpath(path, "$(c)_graphics"))
    # args["viz"] && visualize_inference(chain, dg, gt_states, gm,
    #                                    joinpath(path, "$(c)_scene"))

    return nothing
end

main();
