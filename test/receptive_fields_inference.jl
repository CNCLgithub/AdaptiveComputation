using MOT
using Gen
using Gen_Compose
using Random
using Lazy
using Images, FileIO, PaddedViews
Random.seed!(2)

using StatProfilerHTML

r_fields = (5, 5)
overlap = 2
n_particles = 10

attention_type = :sensitivity
#attention_type = :uniform
attention_sweeps = 20
attention_k = 0.05
attention_x0 = 0

attention_smoothness = 0.1
attention_ancestral_steps = 3
attention_samples = 10

# genearate data
k = 2
gm = GMMaskParams(gauss_r_multiple = 4.0,
                  gauss_std = 0.5,
                  gauss_amp = 0.8,
                  fmasks = true,
                  fmasks_n = 20,
                  img_height = 60,
                  img_width = 60,
                  n_trackers = 8,
                  distractor_rate = 8.0)

motion = ISRDynamics()
prob_threshold = 0.01
scene_data = dgp(k, gm, motion; generate_masks = true)

if attention_type == :sensitivity
    attention = MOT.load(MapSensitivity, "/project/scripts/inference/exp1/td.json", sweeps=attention_sweeps,
                         smoothness = attention_smoothness,
                         x0 = attention_x0,
                     objective=MOT.target_designation_receptive_fields,
                     k = attention_k,
                     ancestral_steps = attention_ancestral_steps,
                     samples = attention_samples)
elseif attention_type == :uniform
    attention = UniformAttention(sweeps = attention_sweeps,
                                 ancestral_steps = attention_ancestral_steps)
else
    attention = nothing
end

proc_json = "/project/scripts/inference/exp1/proc.json"
path = joinpath("output", "experiments", "receptive_fields", "test")
try
    mkpath(path)
catch
end

masks = scene_data[:masks][2:end]
gt_causal_graphs = scene_data[:gt_causal_graphs]

# using inertia motion model for inference
motion_inference = InertiaModel(vel = 10.0,
                                low_w = 0.001,
                                high_w = 3.5,
                                a = 0.1,
                                b = 0.4)


# inference prep
latent_map = MOT.LatentMap(Dict(
                            :causal_graph => MOT.extract_causal_graph,
                            #:assignments => MOT.extract_assignments_receptive_fields,
                            :rfs_vec => MOT.extract_rfs_vec
                           ))
latent_map_end = MOT.LatentMap(Dict(
                                :assignments => MOT.extract_assignments_receptive_fields
                               ))

constraints = Gen.choicemap()
init_dots = gt_causal_graphs[1].elements
for i=1:gm.n_trackers
    addr = :init_state => :trackers => i => :x
    constraints[addr] = init_dots[i].pos[1]
    addr = :init_state => :trackers => i => :y
    constraints[addr] = init_dots[i].pos[2]
end
    
# crop observations into receptive fields
receptive_fields = get_rectangle_receptive_fields(r_fields..., gm, overlap = overlap)

args = [(t, motion_inference, gm, receptive_fields, prob_threshold) for t in 1:k]
observations = Vector{Gen.ChoiceMap}(undef, k)
for t = 1:k
    cm = Gen.choicemap()
    
    cropped_masks = @>> receptive_fields map(rf -> MOT.cropfilter(rf, masks[t]))
    # counting the non-zero masks in the timestep
    #display(@>> cropped_masks map(length) maximum)

    for i=1:length(receptive_fields)
        cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
    end
    observations[t] = cm
end
   
query = Gen_Compose.SequentialQuery(latent_map,
                                    latent_map_end,
                                    gm_receptive_fields,
                                    (0, motion_inference, gm, receptive_fields, prob_threshold),
                                    constraints,
                                    args,
                                    observations)

proc = MOT.load(PopParticleFilter, proc_json;
                particles = n_particles,
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))

@time results = sequential_monte_carlo(proc, query,
                                 buffer_size = k,
                                 path = joinpath(path, "results.jld2"))


visualize_inference(results, gt_causal_graphs,
                    gm, attention, dirname(path),
                    receptive_fields = r_fields,
                    receptive_fields_overlap = overlap)


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

MOT.get_target_designation(gm.n_trackers, map_assignment, masks[end], receptive_fields)
