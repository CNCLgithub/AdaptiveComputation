using MOT
using Gen
using Random
Random.seed!(1)

k = 120
motion = HGMDynamicsModel()
cm = Gen.choicemap()
scene_data = nothing
tries = 0

# while true
    # tries += 1
    # print("tries: $tries \r")
    # scene_data = dgp(k, default_hgm, motion;
                     # generate_masks=false,
                     # cm=cm)
    # is_min_distance_satisfied(scene_data, 80.0) && break
# end
scene_data = dgp(k, default_hgm, motion;
                 generate_masks=false,
                 cm=cm)

render(default_gm, k;
       gt_causal_graphs=scene_data[:gt_causal_graphs],
       freeze_time=24,
       stimuli=true)
