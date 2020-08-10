using MOT
using Random

Random.seed!(0)

q = ISRDynamicsExperiment(k=60)
run_inference(q, "out", viz=true)
