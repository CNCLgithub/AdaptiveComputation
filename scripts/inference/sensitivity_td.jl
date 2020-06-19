using MOT

function main()
    exp = SensTDExperiment()
    trial_idx = 0
    out = "/experiments/$(get_name(exp))/$trial_idx"
    run_inference(exp, out)
end


main()
