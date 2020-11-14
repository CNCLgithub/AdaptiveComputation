using CSV
using MOT

args = Dict("gm" => "/project/scripts/inference/isr_inertia/gm.json",
            "dataset" => "/datasets/exp1_isr_480.jld2",
            "scene" => 2,
            "time" => 1,
            "proc" => "/project/scripts/inference/isr_inertia/proc.json",
            "motion" => "/project/scripts/inference/isr_inertia/motion.json",
            "chain" => 1,
            "target_designation" => Dict("params" => "/project/scripts/inference/isr_inertia/td.json"),
            "viz" => false,
            "restart" => true)

experiment_name = "isr_inertia"

function test_isr_inertia(args, att_mode)

   att = MOT.load(MapSensitivity, args[att_mode]["params"])

   motion = MOT.load(InertiaModel, args["motion"])

   query, gt_causal_graphs, gm_params = query_from_params(args["gm"], args["dataset"],
                                                         args["scene"], args["time"],
                                                         gm = gm_inertia_mask,
                                                         motion = motion)

   proc = MOT.load(PopParticleFilter, args["proc"];
                  rejuvenation = rejuvenate_attention!,
                  rejuv_args = (att,))

   base_path = "/experiments/$(experiment_name)_$(att_mode)"
   scene = args["scene"]
   path = joinpath(base_path, "$(scene)")
   try
      isdir(base_path) || mkpath(base_path)
      isdir(path) || mkpath(path)
   catch e
      println("could not make dir $(path)")
   end
   c = args["chain"]
   out = joinpath(path, "$(c).jld2")
   if isfile(out) && !args["restart"]
      println("chain $c complete")
      return
   end

   println("running chain $c")
   results = run_inference(query, proc, out)

   if (args["viz"])
      visualize_inference(results, gt_causal_graphs, gm_params, att, path;
                           render_tracker_masks=true)
   end
   df = MOT.analyze_chain(results)
   df[!, :scene] .= args["scene"]
   df[!, :chain] .= c
   CSV.write(joinpath(path, "$(c).csv"), df)
   return df
end


results = test_isr_inertia(args, "target_designation")
