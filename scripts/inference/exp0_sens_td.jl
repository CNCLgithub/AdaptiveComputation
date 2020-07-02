using MOT
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--chains", "-c"
        help = "The number of chains to run"
        arg_type = Int
        default = 1

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "trial"
        help = "Which trial to run"
        arg_type = Int
        required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    exp = Exp0SensTD(;trial = args["trial"])
    path = "/experiments/$(get_name(exp))/$(exp.trial)"
    isdir(path) || mkpath(path)
    for c = 1:args["chains"]
        out = joinpath(path, "$c")
        if isfile(joinpath(out, "trace.jld")) && !args["restart"]
            continue
        end
        run_inference(exp, out)
    end
    return nothing
end


main();
