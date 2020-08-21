using MOT
using Random
using JLD2

k = 5
Random.seed!(1234)
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath("datasets_folder")
dataset_path = joinpath(datasets_folder, "isr_dataset.jld2")

q = ISRDynamicsExperiment(k=k)
gm = MOT.load(GMMaskParams, q.gm)
motion = MOT.load(ISRDynamics, q.motion)

println("generating dataset")
jldopen(dataset_path, "w") do file 
    for i=1:1
        # generating initial positions and masks (observations)
        init_positions, masks, positions = dgp(q.k, gm, motion)

        trial = JLD2.Group(file, "$i")
        trial["motion"] = motion
        trial["positions"] = positions
        trial["init_positions"] = init_positions
    end
end

q = ISRDynamicsExperiment(k=k, trial=1)
att = MapSensitivity()
path = "/experiments/test"
run_inference(q, att, path)
