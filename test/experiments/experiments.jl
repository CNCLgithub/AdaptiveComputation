exp = Exp0(;k = 10)
att = MapSensitivity()
path = "/experiments/test/test.jld2"
ispath("/experiments/test") || mkdir("")
run_inference(exp, att, path; viz = true)
