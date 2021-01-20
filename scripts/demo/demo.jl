using MOT
using Random
Random.seed!(4)

#### GENERATING THE DATASET

n_scenes = 2 # number of scenes in the dataset
T = 240 # number of timesteps in one scene
min_distance = 80

# specifying the output path for the dataset (saves using JLD2 as an HDF5 file)
dataset_file = "isr_dataset.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

# you can specify the motion model to use here
motion = ISRDynamics()

println("generating ISR dataset...")
MOT.generate_dataset(dataset_path, n_scenes, T, default_gm_params, motion;
                     min_distance = min_distance)
println("generating ISR dataset done. written to $dataset_path")


#### RENDERING THE DATASET
render_path = joinpath("output", "renders", "isr_dataset")
println("rendering dataset...")
MOT.render_dataset(dataset_path, render_path, default_gm_params)
println("rendering dataset done.")
