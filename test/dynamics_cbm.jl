using MOT


q = ExampleExperiment(k=120)
gm_params = GMMaskParams(init_vel=5.0)
motion = ConstrainedBDM()

init_positions, masks, positions = dgp(q.k, gm_params, motion, dynamics="cbm")

render(positions, gm_params)
