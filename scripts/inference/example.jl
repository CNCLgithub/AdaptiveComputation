using MOT

function main()
    exp = ExampleExperiment()
    trial_idx = 0
    out = "/experiments/$(get_name(exp))/$trial_idx"
    run_inference(exp)
end


main()
