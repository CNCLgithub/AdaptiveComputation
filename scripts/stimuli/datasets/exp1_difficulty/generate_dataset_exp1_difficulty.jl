using MOT
using MOT: @set, choicemap
using Random
Random.seed!(4)

k = 100

dataset_file = "exp1_difficulty.jld2"
datasets_folder = joinpath("output", "datasets")
ispath(datasets_folder) || mkpath(datasets_folder)
dataset_path = joinpath(datasets_folder, dataset_file)

#main_gm = MOT.load(GMParams, "$(@__DIR__)/gm.json")
#main_dm = MOT.load(ISRDynamics, "$(@__DIR__)/dm.json")

main_gm = HGMParams(n_trackers = 4,
                    area_height = 800,
                    area_width = 800)

main_dm = SquishyDynamicsModel(poly_rep_m = 80.0,
                               poly_rep_a = 0.07,
                               poly_rep_x0 = 0.0,
                               wall_rep_m = 20.0,
                               wall_rep_a = 0.04,
                               wall_rep_x0 = 0.0,
                               poly_att_m = 1.0,
                               poly_att_a = 0.05,
                               poly_att_x0 = 0.0)

# dimension of difficulty: velocity and number of distractors
vels = LinRange(5.0, 15.0, 5)
n_distractors = collect(3:7)

vels = [5.0, 15.0]
n_distractors = [3, 7]

#vels = [15.0]
#n_distractors = [7]

n_scenes_per_pair = 1
n_scenes = length(vels) * length(n_distractors) * n_scenes_per_pair

gms = Vector{HGMParams}(undef, n_scenes)
dms = Vector{SquishyDynamicsModel}(undef, n_scenes)
cms = Vector{MOT.ChoiceMap}(undef, n_scenes)
aux_data = Vector{Any}(undef, n_scenes)
ff_ks = fill(3, n_scenes)

for (i, vn) in enumerate(Iterators.product(vels, n_distractors, 1:n_scenes_per_pair))
    vel, n_dist, _ = vn

    # making a copy of the generative model and dynamics model parameters
    gm = deepcopy(main_gm)
    dm = deepcopy(main_dm)
    
    # adjusting based on particular scene
    gm = @set gm.distractor_rate = n_dist
    targets = [fill(1, gm.n_trackers); fill(0, n_dist)]
    gms[i] = @set gm.targets = targets
    dms[i] = @set dm.vel = vel
    
    cm = choicemap()
    MOT.@>> 1:n_dist+gm.n_trackers begin
        foreach(i -> cm[:init_state => :polygons => i => :n_dots] = 1)
    end
    cms[i] = cm

    # adding auxiliary data about how the scene is structured for
    # rendering purposes (on psiturk and offline)
    aux_data[i] = (scene_structure = fill(1, n_dist + gm.n_trackers),
                   targets = targets,
                   vel = vel,
                   n_dist = n_dist)
end

println("generating exp1 difficulty dataset...")
MOT.generate_dataset(dataset_path, n_scenes, k, gms, dms, cms=cms, ff_ks=ff_ks, aux_data=aux_data)
println("generating exp1 difficulty dataset done. written to $dataset_path")

include("../convert_dataset_to_json.jl")
convert_dataset_to_json("output/datasets/exp1_difficulty.jld2", "output/datasets/exp1_difficulty.json")
run(`cp output/datasets/exp1_difficulty.json ../mot-psiturk/psiturk/static/data/dataset.json`)
include("render.jl")
