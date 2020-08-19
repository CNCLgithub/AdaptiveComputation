exp = Exp0(;k = 10, trial = 124)
att = MapSensitivity(samples = 5,
                     sweeps = 15,
                     k = 1.16,
                     x0 = 3.0,
                     objective = MOT.data_correspondence)
path = "/experiments/test/test.jld2"
ispath("/experiments/test") || mkdir("/experiments/test")
run_inference(exp, att, path; viz = true)
