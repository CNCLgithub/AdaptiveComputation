using MOT
using Gen
using Gen_Compose
using Random
using Lazy
using Statistics
using Images, FileIO, PaddedViews
Random.seed!(1)

using StatProfilerHTML

r_fields = (4, 4)
overlap = 3
n_particles = 15

attention_type = :sensitivity
attention_sweeps = 10
attention_k = 0.03
attention_x0 = 20

attention_smoothness = 0.05
attention_ancestral_steps = 3
attention_samples = 10

# genearate data
k = 80
gm = GMMaskParams(gauss_r_multiple = 8.0,
                  gauss_std = 0.5,
                  gauss_amp = 0.8,
                  fmasks = true,
                  fmasks_n = 8,
                  img_height = 50,
                  img_width = 50,
                  n_trackers = 4,
                  distractor_rate = 4.0)

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
function cropfilter(rf, masks)
    cropped_masks = map(mask -> MOT.crop(rf, mask), masks)
    croppedfiltered_masks = filter(mask -> any(mask .!= 0), cropped_masks)
end

args = [(t, motion_inference, gm, receptive_fields, prob_threshold) for t in 1:k]
observations = Vector{Gen.ChoiceMap}(undef, k)
for t = 1:k
    cm = Gen.choicemap()
    
    cropped_masks = @>> receptive_fields map(rf -> cropfilter(rf, masks[t]))
    # counting the non-zero masks in the timestep
    #display(@>> cropped_masks map(length) maximum)

    for i=1:length(receptive_fields)
        cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
    end
    observations[t] = cm
end
   
query = Gen_Compose.SequentialQuery(latent_map,
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

