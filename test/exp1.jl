using MOT
using Random
Random.seed!(1)

# gm = GMMaskParams()
# init_positions, masks, motion, positions = load_trial(1, "/datasets/exp1.jld2", gm)
# render(gm, dot_positions=positions)
# error()

q = Exp1(trial=5,
         k=120)
attention = MapSensitivity(samples=3,
                           sweeps=10,
                           smoothness=1.005,
                           objective=MOT.data_correspondence)
path = "/experiments/test"
mkpath(path)

run_inference(q, attention, path, viz=true)
