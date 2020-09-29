using MOT
using Random
Random.seed!(1)

motion = ISRDynamics()
k = 120
trial_data = dgp(k, default_gm, motion; generate_masks=false)
render(default_gm, k; gt_causal_graphs=trial_data[:gt_causal_graphs])

