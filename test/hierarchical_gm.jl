using MOT
using Random

Random.seed!(0)

k = 120
params = GMHMaskParams(polygon_radius = 100.0, init_pos_spread=200.0)
hbmm = HierarchicalBrownianDynamicsModel()

_, _, masks, positions = dgp(k, params, hbmm)
render(positions, params)
