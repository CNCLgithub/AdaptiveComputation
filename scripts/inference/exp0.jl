using MOT
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "--viz", "-v"
        help = "Whether to render masks"
        action = :store_true

        "trial"
        help = "Which trial to run"
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

        "trial_avg", "A"
        help = "Using trial avg"
        action = :command

    end


    @add_arg_table! s["target_designation"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/exp0/td.json"

    end

    @add_arg_table! s["data_correspondence"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/exp0/dc.json"
    end
    @add_arg_table! s["trial_avg"] begin
        "model_path"
        help = "path containing compute allocations"
        arg_type = String
        required = true

    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    exp = Exp0(;trial = args["trial"], k = 120)
    att_mode = args["%COMMAND%"]
    if att_mode == "target_designation"
        att = load(MapSensitivity, args[att_mode]["params"])
    elseif att_mode == "data_correspondence"
        att = load(MapSensitivity, args[att_mode]["params"];
                   objective = MOT.data_correspondence)
    else
        att = load(UniformAttention, args[att_mode]["model_path"],
                   exp.trial, exp.k)
    end

    path = "/experiments/$(get_name(exp))_$(att_mode)/$(exp.trial)"
    isdir(path) || mkpath(path)
    c = args["chain"]
    out = joinpath(path, "$(c).jld2")
    if isfile(out) && !args["restart"]
        println("chain $c complete")
        return
    end
    println("running chain $c")
    run_inference(exp, att, out; viz = args["viz"])
    return nothing
end


main();
