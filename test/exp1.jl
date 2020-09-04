using MOT
using Random
Random.seed!(1)

# gm = GMMaskParams()
# init_positions, masks, motion, positions = load_trial(1, "/datasets/exp1.jld2", gm)
# render(gm, dot_positions=positions)
# error()

q = Exp1ISR(trial=5,
            k= 60)
attention = MapSensitivity(samples=5,
                           sweeps=20,
                           smoothness=0.01,
                           k = 3350.,
                           x0 = 1.68E11,
                           scale = 495.,
                           # k = 1.0009,
                           # x0 = 1200.0,
                           # scale = 20.0,
                           objective=MOT.target_designation)
path = "/experiments/test"
mkpath(path)

run_inference(q, attention, path, viz=true)
