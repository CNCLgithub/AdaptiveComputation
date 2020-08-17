using MOT
using Random

Random.seed!(1234)

q = ISRDynamicsExperiment(k=30)

path = "/experiments/test/test.jld2"
ispath("/experiments/test") || mkdir("/experiments/test")

att = MapSensitivity()
gm_params = MOT.load(GMMaskParams, q.gm)
motion = MOT.load(ISRDynamics, q.motion)

# generating initial positions and masks (observations)
init_positions, masks, positions = dgp(q.k, gm_params, motion)
render(positions, gm_params;
        path = joinpath(path, "render"))

run_inference(q, att, path; viz = true)
