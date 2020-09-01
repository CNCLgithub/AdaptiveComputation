using MOT
using Random

Random.seed!(3)

q = Exp1(k=120, trial=1,
         gm="scripts/inference/exp1/gm.json",
         proc="scripts/inference/exp1/proc.json",
         dataset_path="output/datasets/exp1_isr.jld2")


path = "/experiments/test"
mkpath(path)

att = MOT.load(MapSensitivity, "scripts/inference/exp1/td.json")
results = run_inference(q, att, path)
println("final assignment: $(extract_chain(results)["unweighted"][:assignments][end,1])")
