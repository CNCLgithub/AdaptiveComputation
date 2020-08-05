using MOT
using Random

Random.seed!(0)

q = ExampleExperiment(k=120)
gm_params = GMMaskParams(init_vel=5.0)
motion = ISRDynamics()

init_positions, masks, positions = dgp(q.k, gm_params, motion, dynamics="isr")

render(positions, gm_params)
