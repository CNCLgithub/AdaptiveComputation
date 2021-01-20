using MOT
using Random
Random.seed!(4)

n_scenes = 60 # number of scenes in the dataset
k = 240 # number of timesteps in one scene

# specifying the output path for the dataset (saves using JLD2 as an HDF5 file)
dataset_file = "isr_dataset.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

# you can specify
motion = ISRDynamics()

println("generating ISR dataset...")
MOT.generate_dataset(dataset_path, n_scenes, k, default_gm, motion)
println("generating ISR dataset done. written to $dataset_path")
