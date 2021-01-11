using MOT
using Gen
using Random
Random.seed!(4)
include("exp3_polygons_structure.jl")
dataset_path = joinpath("/datasets", "exp3_polygons.jld2")
n_scenes_2_generate = 20000 # generate and remove non-unique scenes
k = 240

function sample_polygon_structure(n_dots_limit::Int)::Vector{Int}
    n_dots_remaining = n_dots_limit
    polygon_structure = Int[]
    while true
        polygon = bernoulli(0.8)
        potential_object = polygon ? uniform_discrete(3, 5) : 1
        if potential_object <= n_dots_remaining
            push!(polygon_structure, potential_object)
            n_dots_remaining -= potential_object
        end
        if n_dots_remaining == 0
            sort!(polygon_structure, rev=true)
            return polygon_structure
        end
    end
end

polygon_structures = map(_ -> sample_polygon_structure(8), 1:n_scenes_2_generate)
#display(polygon_structures)

motion = HGMDynamicsModel()
cms = Vector{ChoiceMap}(undef, n_scenes_2_generate)
for i=1:n_scenes_2_generate
    cm = Gen.choicemap()
    for (j, object) in enumerate(polygon_structures[i])
        cm[:init_state => :trackers => j => :polygon] = object > 1
        if object > 1
            cm[:init_state => :trackers => j => :n_dots] = object
        end
    end
    cms[i] = cm
end


# this is the default target designation that we permute
# for each trial to generate different target concentrations
normal_targets = Bool[1, 1, 1, 1, 0, 0, 0, 0]

function describe_scene(i)
    polygon_structure = polygon_structures[i]
    targets = shuffle(normal_targets)
    structure_value = 1/length(polygon_structure)
    target_concentration = get_target_concentration(polygon_structure, targets)

    Dict([:polygon_structure => polygon_structure,
          :targets => targets,
          :structure_value => structure_value,
          :target_concentration => target_concentration])
end

"""
Tests whether there are any target-only or distractor-only polygons
that appear only once. We want to eliminate those scenes because
no tracking is needed in those cases.
"""
function polygon_concentration_condition(scene)
    pols = scene[:polygon_structure]
    targets = scene[:targets]
    
    # looking for fully concentrated polygons
    fc_pols_targets = []
    fc_pols_distractors = []

    index = 1
    for pol in pols
        if pol == 1
            index += 1
            continue
        else
            pol_targets = targets[index:index+pol-1]
            ptc = sum(pol_targets)/length(pol_targets)
            if ptc == 1
                push!(fc_pols_targets, pol)
            elseif ptc == 0
                push!(fc_pols_distractors, pol)
            end
            index += pol
        end
    end
    
    fc_pols_targets = sort(fc_pols_targets, rev=true)
    fc_pols_distractors = sort(fc_pols_distractors, rev=true)
    
    # we want these to match -- that means for every fully concentrated
    # target-only polygon there is counterbalancing distractor-only polygon
    return fc_pols_targets == fc_pols_distractors
end

# we store polygon structure and targets in the aux_data as a Vector{Dict}
aux_data = map(i -> describe_scene(i), 1:n_scenes_2_generate)
aux_data = convert(Vector{Any}, aux_data)

# we make sure that
pol_passed = map(polygon_concentration_condition, aux_data)
aux_data = aux_data[pol_passed]
cms = cms[pol_passed]

# we don't care about targets vector uniqueness
values = map(x -> (x[:polygon_structure], x[:structure_value], x[:target_concentration]), aux_data)

unique_scenes_idxs = findfirst.(isequal.(unique(values)), [values])
n_unique_scenes = length(unique_scenes_idxs)
println("NUMBER OF UNIQUE SCENES GENERATED: ", n_unique_scenes)

cms = cms[unique_scenes_idxs]
aux_data = aux_data[unique_scenes_idxs]

gms = Vector{HGMParams}(undef, n_unique_scenes)
for i=1:n_unique_scenes
    n_trackers = length(aux_data[i][:polygon_structure])
    gms[i] = HGMParams(n_trackers = n_trackers,
                       distractor_rate = 0.0,
                       targets = aux_data[i][:targets])
end

MOT.generate_dataset(dataset_path, n_unique_scenes, k, gms, motion;
                     min_distance = 50.0,
                     cms=cms,
                     aux_data=aux_data)

