using MOT
using Random
Random.seed!(2)

scene = 1
k = 60
fmasks_decay_function=x->x/1.2
fmasks_n = 5

q = Exp1(scene=scene, k=k,
         fmasks=true,
         fmasks_decay_function=fmasks_decay_function,
         fmasks_n=fmasks_n)

att = MapSensitivity(samples=3,

                     sweeps=20,
                     smoothness=1.007,
                     k = 3350.,
                     x0 = 1.68E11,
                     scale = 495.,
                     objective=MOT.target_designation)


path = "/experiments/test_flow_masks/"
mkpath(path)
out = joinpath(path, "results.jld2")
run_inference(q, att, out, viz=true);
