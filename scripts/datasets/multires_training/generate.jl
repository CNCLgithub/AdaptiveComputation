using Gen
using MOT
using MOT: @set, choicemap
using JLD2
using Random

# Set seed for reproducability
Random.seed!(1234)


# declare dataset io paths
dataset_name = "mutlires_training.jld2"
datasets_path = "/spaths/datasets/$dataset_name"


# load parameters
gm = MOT.load(GMParams, "$(@__DIR__)/gm.json") # number of objects etc.
dm = MOT.load(ISRDynamics, "$(@__DIR__)/dm.json") # motion dynamcis
gr = MOT.load(Graphics, "") # load graphics / visuals

# dimension of difficulty: velocity and number of distractors
vels = LinRange(2.0, 14.0, 13) # 2 -> 14.0, with 13 steps
# a tracker is "individual object" or a MOT.Dot
n_trackers = collect(2:8) #

n_scenes_per_pair = 1
n_scenes = length(vels) * length(n_distractors) * n_scenes_per_pair

i = 1
for (v, nt) in Iterators.product(vels, n_distractors)
    # generate returns a (Gen.Trace, log score)
    trace, _ = Gen.generate(gm_isr_pos, ...) # TODO; define args
    # (CausalGraph, Vector{CausalgeGraph})
    (init_state, states) = ... # get the return value from `trace`

    # TODO invoke dataset
    # check out https://juliaio.github.io/JLD2.jl/dev/#A-new-interface:-jldsave
    # dataset[i => :init_state] = init_state
    # dataset[i => :states] = states

    i += 1
end
