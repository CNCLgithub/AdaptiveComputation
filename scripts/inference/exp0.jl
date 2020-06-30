using MOT
using Random

function main()
    Random.seed!(0)

    exp = Exp0(trial = 124)
    results = run_inference(exp)
    return results
end

main()
