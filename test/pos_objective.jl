using MOT
using Random
Random.seed!(1)

q = Exp0(scene=124,
         k=120)
attention = MapSensitivity(samples=5,
                           sweeps=20,
                           smoothness=0.01,
                           k = 3350.,
                           x0 = 1.68E11,
                           scale = 495.,
                           objective=MOT.pos_objective)
path = "/experiments/test"
mkpath(path)

results = run_inference(q, attention, joinpath(path, "results.jld2"), viz=true)
