using MOT

function main()
    exp = ExampleExperiment()
    out = "/experiments/$(name(exp))/$trial_idx"
    run_inference(exp)
end


main()
