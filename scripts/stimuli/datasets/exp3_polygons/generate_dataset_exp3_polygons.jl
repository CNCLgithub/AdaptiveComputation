using MOT
using Gen
using Random
Random.seed!(4)

dataset_path = joinpath("/datasets", "exp3_polygons.jld2")
k = 240
motion = HGMDynamicsModel()
cm = Gen.choicemap()
cm[:init_state => :trackers => 1 => :polygon] = true
cm[:init_state => :trackers => 1 => :n_dots] = 4
cm[:init_state => :trackers => 2 => :polygon] = true
cm[:init_state => :trackers => 2 => :n_dots] = 3
cm[:init_state => :trackers => 3 => :polygon] = false
targets = Bool[1, 1, 1, 0, 1, 0, 0, 0]

n_scenes = 20
cms = convert(Vector{ChoiceMap}, fill(cm, n_scenes))
aux_data = convert(Vector{Any}, fill(targets, n_scenes))

hgm = HGMParams(n_trackers = 3,
                distractor_rate = 0.0,
                targets = [1, 1, 0, 0])

MOT.generate_dataset(dataset_path, n_scenes, k, hgm, motion;
                     min_distance = 100.0,
                     cms=cms,
                     aux_data=aux_data)

