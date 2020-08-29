using MOT
using Random
Random.seed!(4)

n_trials = 10
k = 120


dataset_file = "exp1.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath("datasets_folder")
dataset_path = joinpath(datasets_folder, dataset_file)

gm_path = "$(@__DIR__)/dataset_exp1/gm.json"
motion_path = "$(@__DIR__)/dataset_exp1/motion.json"

gm = MOT.load(GMMaskParams, gm_path)
motion = MOT.load(BrownianDynamicsModel, motion_path)

println("generating exp1 dataset...")
MOT.generate_dataset(dataset_path, n_trials, k, gm, motion)
println("generating exp1 dataset done. written to $dataset_path")
