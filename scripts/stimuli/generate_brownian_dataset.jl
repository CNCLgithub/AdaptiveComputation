using MOT
using Random
Random.seed!(4)

n_trials = 2
k = 120

dataset_file = "brownian_dataset.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath("datasets_folder")
dataset_path = joinpath(datasets_folder, dataset_file)

gm_path = "$(@__DIR__)/brownian_dataset/gm.json"
motion_path = "$(@__DIR__)/brownian_dataset/motion.json"

gm = MOT.load(GMMaskParams, gm_path)
motion = MOT.load(BrownianDynamicsModel, motion_path)

MOT.generate_dataset(dataset_path, n_trials, k, gm, motion)
