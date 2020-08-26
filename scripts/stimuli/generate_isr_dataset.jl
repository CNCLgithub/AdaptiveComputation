using MOT
using Random
using JLD2

n_trials = 20

k = 120
Random.seed!(4)
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath("datasets_folder")
dataset_path = joinpath(datasets_folder, "isr_dataset.jld2")

q = ISRDynamicsExperiment(k=k)
gm = MOT.load(GMMaskParams, q.gm)
motion = MOT.load(ISRDynamics, q.motion)

println("generating dataset")
jldopen(dataset_path, "w") do file 
    file["n_trials"] = n_trials
    for i=1:n_trials
        # generating initial positions and masks (observations)
        init_positions, init_vels, masks, positions = dgp(q.k, gm, motion)
        # render(gm, dot_positions=positions, stimuli=true, freeze_time=30, highlighted=collect(1:4))

        trial = JLD2.Group(file, "$i")
        trial["motion"] = motion
        trial["positions"] = positions
        trial["init_positions"] = init_positions
        trial["gm"] = gm
    end
end

q = ISRDynamicsExperiment(k=k, trial=1, motion="motion.json")
att = MapSensitivity(samples=5)
path = "/experiments/test"
run_inference(q, att, path)
