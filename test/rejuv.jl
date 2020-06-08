using MOT
using Gen

Gen.load_generated_functions()
(trace, w) = Gen.generate(generative_model, (5, default_params))
choices = Gen.get_choices(trace)
println(choices)

varied = collect(1:default_params.num_observations)
ps = fill(1.0/default_params.num_observations,
          default_params.num_observations)
args = (varied, ps)
(trace, a) = object_move(trace, args)
choices = Gen.get_choices(trace)
println(choices)
println(a)
