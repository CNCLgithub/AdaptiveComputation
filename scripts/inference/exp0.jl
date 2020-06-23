using MOT
using Random

function main()
    Random.seed!(0)

    exp = Exp0(trial = 124)
    out = "/experiments/$(get_name(exp))/$(exp.trial)"
    results = run_inference(exp)
    return results
end

#main()
