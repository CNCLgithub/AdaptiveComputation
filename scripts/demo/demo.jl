using MOT
using Random
Random.seed!(4)


#### SPECIFYING PARAMETERS FOR MOTION AND GENERATIVE MODEL/GRAPHICS
n_scenes = 2 # number of scenes in the dataset
T = 240 # number of timesteps in one scene
min_distance = 50 # enforcing minimum distance between dots at the beggining of the movement

# you can specify the motion model and its parameters
# (parameters not specified are instantiated with default values)
motion = ISRDynamics(dot_repulsion = 10.0,
                     wall_repulsion = 50.0,
                     vel = 100.0)
# or load the motion parameters from file
motion_file = joinpath("$(@__DIR__)", "motion.json")
motion = MOT.load(ISRDynamics, motion_file)

# similarly for the generative model parameters
gm_params = GMParams(n_trackers = 8,
                     distractor_rate = 9,
                     dot_radius = 20.0,
                     area_height = 800.0,
                     area_width = 800.0)

# or load from file
#gm_params_file = joinpath("$(@__DIR__)", "gm_params.json")
#gm_params = MOT.load(ISRDynamicsm, gm_params_file)


#### DATASET GENERATION
# specifying the output path for the dataset (saves using JLD2 as an HDF5 file)
dataset_file = "isr_dataset.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

println("generating ISR dataset...")
MOT.generate_dataset(dataset_path, n_scenes, T, gm_params, motion;
                     min_distance = min_distance)
println("generating ISR dataset done. written to $dataset_path")


#### RENDERING THE DATASET
render_path = joinpath("output", "renders", "isr_dataset")
println("rendering dataset...")
MOT.render_dataset(dataset_path, render_path, gm_params;
                   stimuli = true,
                   freeze_time = 24,
                   highlighted_start = collect(1:6),
                   highlighted_end = collect(1:3))
println("rendering dataset done.")
