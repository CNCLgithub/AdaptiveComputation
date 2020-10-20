using MOT
using Random
Random.seed!(4)

n_scenes = 60
k = 192

dataset_file = "exp1_isr_extended.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

# motion = ISRDynamics()
scene_data= MOT.load_scene(1, joinpath(datasets_folder, "exp1_isr.jld2"), default_gm;
                           generate_masks=false)

motion = scene_data[:motion]

println("generating exp1 ISR dataset...")
MOT.generate_dataset(dataset_path, n_scenes, k, default_gm, motion)
println("generating exp1 ISR dataset done. written to $dataset_path")
