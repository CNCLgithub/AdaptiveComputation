using MOT
using Gen
using Gen_Compose

using Random
Random.seed!(2)
using Lazy
using StatProfilerHTML
using Images, FileIO, PaddedViews

r_fields = (5, 5)
overlap = 2
n_particles = 15

# load data
hgm = MOT.load(HGMParams, joinpath("$(@__DIR__)", "hgm.json"))
scene_data = MOT.load_scene(2, joinpath("/datasets", "exp3_polygons.jld2"), hgm)

k = 80
prob_threshold = 0.01

attention = MOT.load(MapSensitivity, joinpath("$(@__DIR__)", "attention.json"),
                     objective=MOT.target_designation_receptive_fields)

#attention = UniformAttention(sweeps=0)

path = joinpath("output", "experiments", "receptive_fields", "test")
try
    mkpath(path)
catch
end

masks = scene_data[:masks][2:end]
gt_causal_graphs = scene_data[:gt_causal_graphs]

# # using inertia motion model for inference
motion_inference = MOT.load(HGMInertiaDynamicsModel, joinpath("$(@__DIR__)", "motion.json"))
display(motion_inference)

# inference prep
latent_map = MOT.LatentMap(Dict(
                            :causal_graph => MOT.extract_causal_graph,
                            :rfs_vec => MOT.extract_rfs_vec
                           ))
latent_map_end = MOT.LatentMap(Dict(
                                :assignments => MOT.extract_assignments_receptive_fields
                               ))


# constraining initial structure
polygon_structure = scene_data[:aux_data][:polygon_structure]
targets = scene_data[:aux_data][:targets]

# getting the indices of hierarchical objects that need to be tracked
trackers = []
targets_in_tracked = []
for i=1:length(polygon_structure)
    j = sum(polygon_structure[1:i-1]) + 1
    k = sum(polygon_structure[1:i])
    # there is at least one target on the polygon
    if any(targets[j:k])
        append!(targets_in_tracked, targets[j:k])
        push!(trackers, i)
    end
end

#targets_in_tracked = ones(8)

println("DISTRACTOR RATE $(sum(targets))")

hgm = MOT.load(HGMParams, joinpath("$(@__DIR__)", "hgm.json"),
               targets = targets_in_tracked,
               n_trackers = length(trackers),
               distractor_rate = sum(targets))

constraints = Gen.choicemap()

# constraining to initial positions
init_objects = gt_causal_graphs[1].elements
println("polygon_structure $polygon_structure")
println("trackers $trackers")
for (i, j) in enumerate(trackers)
    n_dots = polygon_structure[j]
    if n_dots > 1
        constraints[:init_state => :trackers => i => :polygon] = true
        constraints[:init_state => :trackers => i => :n_dots] = n_dots
    else
        constraints[:init_state => :trackers => i => :polygon] = false
    end

    addr = :init_state => :trackers => i => :x
    constraints[addr] = init_objects[j].pos[1]
    addr = :init_state => :trackers => i => :y
    constraints[addr] = init_objects[j].pos[2]

    if init_objects[j] isa Polygon
        addr = :init_state => :trackers => i => :rot
        constraints[addr] = init_objects[j].rot
        pol_dots = init_objects[j].dots
        for k=1:length(pol_dots)
            addr = :init_state => :trackers => i => k => :x
            constraints[addr] = pol_dots[k].pos[1]
            addr = :init_state => :trackers => i => k => :y
            constraints[addr] = pol_dots[k].pos[2]
        end
    end
end

display(constraints)

# crop observations into receptive fields
receptive_fields = get_rectangle_receptive_fields(r_fields..., hgm, overlap = overlap)
args = [(t, motion_inference, hgm, receptive_fields, prob_threshold) for t in 1:k]
observations = Vector{Gen.ChoiceMap}(undef, k)
for t = 1:k
    cm = Gen.choicemap()
    
    cropped_masks = @>> receptive_fields map(rf -> MOT.cropfilter(rf, masks[t]))

    for i=1:length(receptive_fields)
        cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
    end
    observations[t] = cm
end
   
query = Gen_Compose.SequentialQuery(latent_map,
                                    latent_map_end,
                                    hgm_receptive_fields,
                                    (0, motion_inference, hgm, receptive_fields, prob_threshold),
                                    constraints,
                                    args,
                                    observations)

proc = MOT.load(PopParticleFilter, joinpath("$(@__DIR__)", "proc.json"),
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))

@time results = sequential_monte_carlo(proc, query,
                                 buffer_size = k,
                                 path = joinpath(path, "results.jld2"))

visualize_inference(results, gt_causal_graphs,
                    hgm, attention, dirname(path),
                    receptive_fields = r_fields,
                    receptive_fields_overlap = overlap,
                    freeze_time = 10,
                    highlighted = findall(x-> x== 1, targets))


# rendering the masks from the receptive field rfs
out_dir = joinpath("output", "experiments", "receptive_fields", "mask_distributions")
ispath(out_dir) && rm(out_dir, recursive=true)
mkpath(out_dir)

rfs_vecs = @>> begin results.buffer
    map(x -> x["unweighted"][:rfs_vec][1,1,:])
    map(x -> reshape(x, r_fields))
end
@>> 1:k foreach(t -> MOT.save_receptive_fields_img(rfs_vecs[t], t, out_dir))

### target designation extraction
log_scores = results.buffer[end]["log_scores"][1,:]
map_assignment = results.buffer[end]["weighted"][:assignments][1,argmax(log_scores),:]
map_assignment = reshape(map_assignment, r_fields)

# @>> 1:length(masks[end]) foreach(i -> save("test_$i.png", masks[end][i]))
td = MOT.get_target_designation(sum(hgm.targets), map_assignment, masks[k], receptive_fields)

pred_targets = zeros(Bool, length(targets))
pred_targets[first(td)[1]] .= 1
accuracy = sum(targets .& pred_targets)/sum(targets)

println("ground truth targets: \n$targets")
println("model prodiction: \n$pred_targets")
println("accuracy: $accuracy")
