# file used to generate a brownian motion dataset
# based on scene 118 from exp0.jld2

using MOT
using Random
Random.seed!(1)

n_scenes = 50
k = 120

# loading motion from exp0 scene 118
scene_data = MOT.load_scene(118, joinpath("/datasets", "exp0.jld2"), default_gm;
                            generate_masks=false)
motion = scene_data[:motion]

# generating the dataset
dataset_path = joinpath("/datasets", "exp1_brownian.jld2")
println("generating exp1 Brownian dataset (based on exp0 scene 118)...")
MOT.generate_dataset(dataset_path, n_scenes, k, default_gm, motion)
println("done")
