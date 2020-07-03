using MOT
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "trial"
        help = "Which trial to run"
        arg_type = Int
        required = true

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    exp = Exp0SensTD(;trial = args["trial"], k = 120)
    path = "/experiments/$(get_name(exp))/$(exp.trial)"
    isdir(path) || mkpath(path)
    c = args["chain"]
    out = joinpath(path, "$(c).jld2")
    if isfile(out) && !args["restart"]
        println("chain $c complete")
        return
    end
    println("running chain $c")
    run_inference(exp, out)
    return nothing
end


main();
