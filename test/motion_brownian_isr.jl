using MOT


gm = MOT.load(GMMaskParams, "src/experiments/isr_dynamics/gm.json")
motion = MOT.load(ISRDynamics, "motion.json")
init_positions, init_vels, masks, positions = dgp(120, gm, motion)
render(gm; dot_positions=positions)
