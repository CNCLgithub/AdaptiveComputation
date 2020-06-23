using MOT
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "run"
            help = "run"
		    arg_type = Int
            required = true
        "attention"
            arg_type = Bool
            required = true
        "trial"
            arg_type = Int
            required = true
        "compute_type"
            arg_type = String
            required = true
    end

    return parse_args(s)
end


function main()
    #args = parse_commandline()

    test = true
    if !test
        #run = args["run"]
        #trial = args["trial"]
        #attention = args["attention"]
        #compute_type = args["compute_type"]
    else
        run = 9
        trial = 2
        attention = true
        compute_type = "attention"
    end

    
    attention_str = attention ? "attention" : "no_attention_$(compute_type)"
    folder = "exp0_results/$(attention_str)/$(trial)"
    mkpath(folder)

    path = "$folder/$run.jld2"
    if ispath(path)
        error("file exists, exiting..")
    end

    if attention
        exp = Exp0Attention(trial=trial, save_path=path)
    else
        if compute_type == "avg"
            exp = Exp0Avg(trial=trial, save_path=path)
        elseif compute_type == "base"
            exp = Exp0Base(trial=trial, save_path=path)
        else
            error("unrecognized compute type")
        end
    end

    run_inference(exp)
end

main()
