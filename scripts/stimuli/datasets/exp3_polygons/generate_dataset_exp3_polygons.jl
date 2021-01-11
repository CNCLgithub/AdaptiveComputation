using MOT
using Gen
using Random
Random.seed!(4)
include("exp3_polygons_structure.jl")
dataset_path = joinpath("/datasets", "exp3_polygons.jld2")
n_scenes_2_generate = 500000 # generate and remove non-unique scenes
k = 192

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




motion = HGMDynamicsModel()


function describe_scene(polygon_structure,
                        alpha = 1.0,
                        beta = 1.0)
    n_dots = sum(polygon_structure)
    n_targets = Int(n_dots/2)
    normal_targets = Bool[fill(1, n_targets); fill(0, n_targets)]
    targets = shuffle(normal_targets)
    structure_value = 1/length(polygon_structure)
    target_concentration = get_target_concentration(polygon_structure, targets)
    
    rel_structure = (alpha .* structure_value + beta .* target_concentration)/(alpha + beta)

    Dict([:polygon_structure => polygon_structure,
          :targets => targets,
          :structure_value => structure_value,
          :target_concentration => target_concentration,
          :rel_structure => rel_structure])
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



# n_scenes specifies how many scenes to cap at
function get_scene_prereqs(n_targets, n_scenes;
                           alpha = 1.0,
                           beta = 1.0)
    polygon_structures = map(_ -> sample_polygon_structure(n_targets*2), 1:n_scenes_2_generate)

    cms = Vector{ChoiceMap}(undef, n_scenes_2_generate*length(n_targets))
    for i=1:n_scenes_2_generate*length(n_targets)
        cm = Gen.choicemap()
        for (j, object) in enumerate(polygon_structures[i])
            cm[:init_state => :trackers => j => :polygon] = object > 1
            if object > 1
                cm[:init_state => :trackers => j => :n_dots] = object
            end
        end
        cms[i] = cm
    end
    
    alphas = fill(alpha, length(polygon_structures))
    betas = fill(beta, length(polygon_structures))

    # we store polygon structure and targets in the aux_data as a Vector{Dict}
    aux_data = map(describe_scene, polygon_structures, alphas, betas)
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
     
    rs = map(d -> d[:rel_structure], aux_data)
    p = sortperm(rs, rev=true)

    # taking top five, bottom five and random from middle
    top = p[1:5]
    bottom = p[end-4:end]
    middle = shuffle(p[6:end-5])[1:(n_scenes-10)]
    scenes = [top; middle; bottom]

    scene_prereqs = Dict([:aux_data => aux_data[scenes],
                          :cms => cms[scenes],
                          :gms => gms[scenes]])
    return scene_prereqs
end

n_targets = [4, 6, 8]
n_scenes_per_nt = 22
aux_data = []
cms = ChoiceMap[]
gms = []
for nt in n_targets
    scene_prereqs = get_scene_prereqs(nt, n_scenes_per_nt;
                                      alpha = 1.0,
                                      beta = 0.25)
    append!(aux_data, scene_prereqs[:aux_data])
    append!(cms, scene_prereqs[:cms])
    append!(gms, scene_prereqs[:gms])
end

MOT.generate_dataset(dataset_path, length(gms), k, gms, motion;
                     min_distance = 60.0,
                     cms=cms,
                     aux_data=aux_data)

