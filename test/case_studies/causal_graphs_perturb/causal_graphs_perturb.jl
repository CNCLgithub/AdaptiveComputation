using MOT
using MOT: CausalGraph
using Lazy: @>>

using Profile
using StatProfilerHTML

experiment_name = "causal_graph_perturb"

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm_isr" => "$(@__DIR__)/dm_isr.json",
                 "dm_inertia" => "$(@__DIR__)/dm_inertia.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "k" => 20,
                 "viz" => true])
    
    # generating some data using the isr dynamics
    gm = MOT.load(GMParams, args["gm"])
    dm_isr = MOT.load(ISRDynamics, args["dm_isr"])
    gt_cgs = dgp(args["k"], dm_isr, gm)
    
    targets = [fill(true, gm.n_trackers); fill(false, Int(gm.distractor_rate))]

    # inference done using inertia dynamics model
    dm_inertia = MOT.load(InertiaModel, args["dm_inertia"])
    graphics = MOT.load(Graphics, args["graphics"])

    query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm,
                              dm_inertia,
                              graphics,
                              length(gt_cgs))
    
    att = MOT.UniformAttention(sweeps = 2,
                               ancestral_steps = 3)
    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    path = "/experiments/$(experiment_name)"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end

    Profile.init(delay = 0.001,
                 n = 10^6)
    # @profilehtml results = run_inference(prof_query, proc)
    # @profilehtml results = run_inference(query, proc)
    results = run_inference(query, proc)

    df = MOT.analyze_chain_receptive_fields(results,
                                            n_trackers = gm.n_trackers,
                                            n_dots = gm.n_trackers + gm.distractor_rate,
                                            gt_cg_end = gt_cgs[end])

    if (args["viz"])
        visualize_inference(results, gt_cgs, gm,
                            graphics, att, path)
    end

    return results
end

results = main();
