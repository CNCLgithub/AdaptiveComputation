using MOT
using MOT: @set, choicemap
using Random
Random.seed!(4)

k = 120

dataset_file = "exp1_difficulty.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

#main_gm = MOT.load(GMParams, "$(@__DIR__)/gm.json")
#main_dm = MOT.load(ISRDynamics, "$(@__DIR__)/dm.json")

main_gm = HGMParams(n_trackers = 4)
main_dm = SquishyDynamicsModel(poly_rep_m = 20.0,
                               poly_rep_a = 0.05,
                               poly_rep_x0 = 0.0,
                               wall_rep_m = 100.0,
                               wall_rep_a = 0.05,
                               wall_rep_x0 = 0.0,
                               poly_att_m = 0.3,
                               poly_att_a = 0.05,
                               poly_att_x0 = 0.0)

# dimension of difficulty: velocity and
#vels = [5.0, 10.0, 15.0, 20.0]
#n_distractors = [3, 4, 5, 6, 7]

vels = [8.0, 12.0]
n_distractors = [3, 7]
n_scenes_per_pair = 1

n_scenes = length(vels) * length(n_distractors) * n_scenes_per_pair
println(n_scenes)

gms = Vector{HGMParams}(undef, n_scenes)
dms = Vector{SquishyDynamicsModel}(undef, n_scenes)
cms = Vector{MOT.ChoiceMap}(undef, n_scenes)
ff_ks = fill(3, n_scenes)

for (i, vn) in enumerate(Iterators.product(vels, n_distractors, 1:n_scenes_per_pair))
    vel, n_dist, _ = vn

    # making a copy of the generative model and dynamics model parameters
    gm = deepcopy(main_gm)
    dm = deepcopy(main_dm)
    
    # adjusting based on particular scene
    gms[i] = @set gm.distractor_rate = n_dist
    dms[i] = @set dm.vel = vel
    
    cm = choicemap()
    MOT.@>> 1:n_dist+gm.n_trackers begin
        foreach(i -> cm[:init_state => :polygons => i => :n_dots] = 1)
    end
    cms[i] = cm
end

println("generating exp1 difficulty dataset...")
MOT.generate_dataset(dataset_path, n_scenes, k, gms, dms, cms=cms, ff_ks=ff_ks)
println("generating exp1 difficulty dataset done. written to $dataset_path")

include("render.jl")
