using MOT
using Random

function main()
    Random.seed!(0)

    exp = Exp0Base(trial = 124)
    results = run_inference(exp)
end

main()
