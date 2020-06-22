using MOT
using Random

function main()
    Random.seed!(0)

    exp = ExampleExperiment()
    trial_idx = 0
    out = "/experiments/$(get_name(exp))/$trial_idx"
    run_inference(exp)
end


main()
