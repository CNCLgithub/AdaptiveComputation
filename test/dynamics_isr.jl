using MOT
using Random

Random.seed!(1234)
path = "/experiments/test/"
mkpath(path)

q = ISRDynamicsExperiment(k=5)
att = MapSensitivity()
run_inference(q, att, path)
