using MOT
using Gen
using Gen_Compose
using Random
using Images, FileIO, PaddedViews
Random.seed!(1)

r_fields = (10, 10)

# genearate data
k = 3
gm = GMMaskParams(gauss_r_multiple = 2.5,
                  n_trackers = 2,
                  distractor_rate = 2.0)
motion = ISRDynamics()
prob_threshold = 0.01
scene_data = dgp(k, gm, motion; generate_masks = true)


#gm="/project/scripts/inference/exp1_isr/gm.json"
attention = MOT.load(MapSensitivity, "/project/scripts/inference/exp1/td.json", sweeps=0)
proc_json = "/project/scripts/inference/exp1/proc.json"
path = joinpath("output", "experiments", "receptive_fields", "test")
try
    mkpath(path)
catch
end

masks = scene_data[:masks][2:end]
gt_causal_graphs = scene_data[:gt_causal_graphs]
motion = scene_data[:motion]

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
receptive_fields = get_rectangle_receptive_fields(r_fields..., gm)

args = [(t, motion, gm, receptive_fields, prob_threshold) for t in 1:k]
observations = Vector{Gen.ChoiceMap}(undef, k)
for t = 1:k
    cm = Gen.choicemap()
    cropped_masks = map(rf -> map(mask -> MOT.crop(rf, mask), masks[t]), receptive_fields)
    for i=1:length(receptive_fields)
        cm[:kernel => t => :receptive_fields => i => :masks] = cropped_masks[i]
    end
    observations[t] = cm
end
   
query = Gen_Compose.SequentialQuery(latent_map,
                                    gm_receptive_fields,
                                    (0, motion, gm, receptive_fields, prob_threshold),
                                    constraints,
                                    args,
                                    observations)

proc = MOT.load(PopParticleFilter, proc_json;
                #rejuvenation = rejuvenate_attention!,
                rejuv_args = (attention,))
    
results = sequential_monte_carlo(proc, query,
                                 buffer_size = k,
                                 path = joinpath(path, "results.jld2"))
    
visualize_inference(results, gt_causal_graphs,
                    gm, attention, dirname(path))

