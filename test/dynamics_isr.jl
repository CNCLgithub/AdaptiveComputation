using MOT
using Random

Random.seed!(3)

q = ISRDynamicsExperiment(k=60, trial=1)

path = "/experiments/test"
mkpath(path)

att = MapSensitivity(samples=1,
                     sweeps=0,
                     k = 0.05,
                     x0 = 17.8)
results = run_inference(q, att, path)
println("final assignment: $(extract_chain(results)["unweighted"][:assignments][end,1])")
