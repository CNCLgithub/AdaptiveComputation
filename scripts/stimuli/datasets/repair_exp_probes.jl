"""
This fixes an issue with the original archive of the
motions used in the probes experiment.

Generates: `datasets/exp_probes.csv`

"""

using Lazy: @>>
using JSON

# Motion for jld2 file
# MOT.ISRDynamics
#   repulsion: Bool true
#   dot_repulsion: Float64 80.0
#   wall_repulsion: Float64 50.0
#   distance: Float64 60.0
#   vel: Float64 10.0
#   rep_inertia: Float64 0.9
#   brownian: Bool true
#   inertia: Float64 0.8
#   spring: Float64 0.002
#   sigma_x: Float64 1.0
#   sigma_y: Float64 1.0

function repair_isr_480()
    n_scenes = 60 #MOT.get_n_scenes(dataset_path)
    aux_data = (targets = Bool[1, 1, 1, 1, 0, 0, 0, 0],
                vel = 10.0,
                n_distractors = 4)
    dataset = JSON.parsefile("/spaths/datasets/exp_probes_raw.json")
    data = []
    for i=1:n_scenes
        scene = Dict()
        scene[:positions] = dataset[i]
        scene[:aux_data] = aux_data
        push!(data, scene)
    end
    open("/spaths/datasets/exp_probes.json", "w") do f
        write(f, json(data))
    end
end
repair_isr_480()
