exp = Exp0(;k = 20, trial = 124)
att = MapSensitivity(samples = 5,
                     smoothness=0.01,
                     k = 1.01,
                     x0 = 100.0,
                     sweeps = 15,
                     scale = 20.0
                     # objective = MOT.data_correspondence,
                     )
path = "/experiments/test/test.jld2"
ispath("/experiments/test") || mkpath("/experiments/test")
run_inference(exp, att, path; viz = true);
