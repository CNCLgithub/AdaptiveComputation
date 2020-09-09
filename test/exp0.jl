using MOT
using Random
Random.seed!(2)

q = Exp0(trial=124,
         k=3)
attention = MapSensitivity(samples=5,
                           sweeps=20,
                           smoothness=0.01,
                           k = 3350.,
                           x0 = 1.68E11,
                           scale = 495.,
                           objective=MOT.target_designation)
path = "/experiments/test"
mkpath(path)

results = run_inference(q, attention, joinpath(path, "results.jld2"), viz=true)
c = extract_chain(results)
println(c["unweighted"][:causal_graph][1].elements)
