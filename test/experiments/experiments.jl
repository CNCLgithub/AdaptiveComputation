exp = Exp0(;k = 120, trial = 92)
att = MapSensitivity(samples = 5,
                     sweeps = 15,
                     k = 0.05,
                     x0 = 17.8,
                     # objective = MOT.data_correspondence,
                     )
path = "/experiments/test/test.jld2"
ispath("/experiments/test") || mkpath("/experiments/test")
run_inference(exp, att, path; viz = true);
