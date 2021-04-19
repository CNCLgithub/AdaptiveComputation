using MOT
using MOT: @set, choicemap
using Random
Random.seed!(4)

k = 480

dataset_file = "exp1_difficulty.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

#main_gm = MOT.load(GMParams, "$(@__DIR__)/gm.json")
#main_dm = MOT.load(ISRDynamics, "$(@__DIR__)/dm.json")

main_gm = HGMParams()
main_dm = SquishyDynamicsModel()

# dimension of difficulty: velocity and
#vels = [5.0, 10.0, 15.0, 20.0]
#n_distractors = [3, 4, 5, 6, 7]
#n_scenes_per_pair = 2

vels = [5.0, 20.0]
n_distractors = [3, 7]
n_scenes_per_pair = 1

n_scenes = length(vels) * length(n_distractors) * n_scenes_per_pair
println(n_scenes)

gms = Vector{HGMParams}(undef, n_scenes)
dms = Vector{SquishyDynamicsModel}(undef, n_scenes)
cms = fill(choicemap(:n_dots => 1), n_scenes)

for (i, vn) in enumerate(Iterators.product(vels, n_distractors, 1:n_scenes_per_pair))
    vel, n_dist, _ = vn

    # making a copy of the generative model and dynamics model parameters
    gm = deepcopy(main_gm)
    dm = deepcopy(main_dm)
    
    # adjusting based on particular scene
    gms[i] = @set gm.n_trackers = round(Int, n_dist * 2)
    dms[i] = @set dm.vel = vel
end

println("generating exp1 difficulty dataset...")
MOT.generate_dataset(dataset_path, n_scenes, k, gms, dms, cms=cms)
println("generating exp1 difficulty dataset done. written to $dataset_path")
