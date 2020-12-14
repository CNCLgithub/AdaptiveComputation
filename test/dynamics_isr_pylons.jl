using MOT
using Random
Random.seed!(1)

motion = ISRPylonsDynamics()
k =  240
trial_data = dgp(k, default_gm, motion;
                 generate_masks=false)

render(default_gm, k;
       gt_causal_graphs=trial_data[:gt_causal_graphs],
       path="render/1/",
       freeze_time=24,
       highlighted=collect(1:4),
       stimuli=true)
