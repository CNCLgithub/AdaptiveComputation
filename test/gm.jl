using Gen

Gen.load_generated_functions()

(trace, w) = Gen.generate(generative_model, (5, default_params))
# (trace, w) = Gen.generate(init_target_map, ([default_params],))
# target = Target(0, 0, 10, 10, false)
# target = [0., 0., 10., 10.]

# (trace, w) = Gen.generate(distractor_map,
#                           ([tuple() for _ in 1:3],))

choices = Gen.get_choices(trace)
println(choices)
