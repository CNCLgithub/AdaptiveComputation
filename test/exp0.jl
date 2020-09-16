using MOT

q = Exp0(k=5)
attention = MapSensitivity()
path = "/experiments/test"
mkpath(path)
results = run_inference(q, attention, joinpath(path, "results.jld2"), viz=true)
