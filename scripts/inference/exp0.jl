using MOT
using Random

function main()
    Random.seed!(0)

    exp = Exp0(trial = 120)
    #trial_idx = 0
    out = "/experiments/$(get_name(exp))/$(exp.trial)"
    return run_inference(exp)
end


#main()
