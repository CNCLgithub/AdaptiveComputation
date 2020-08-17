using MOT
using Random

Random.seed!(1234)


q = ISRDynamicsExperiment(k=120)
gm = MOT.load(GMMaskParams, q.gm)
motion = MOT.load(ISRDynamics, q.motion)

n_dots = round(Int, gm.n_trackers + gm.distractor_rate)
probes = zeros(Bool, q.k, n_dots)
probes[55:59, 4] .= true

@show probes

# generating initial positions and masks (observations)
init_positions, masks, positions = dgp(q.k, gm, motion)

render(gm;
       dot_positions=positions,
       probes = probes,
       stimuli = true,
       freeze_time = 30)
