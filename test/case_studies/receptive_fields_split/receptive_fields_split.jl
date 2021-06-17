using MOT
using MOT: CausalGraph
using LightGraphs: SimpleDiGraph, add_vertex!
using MetaGraphs: set_prop!, get_prop
using Lazy: @>, @>>

using Random
Random.seed!(1)

experiment_name = "receptive_fields_split"

function get_split_cgs()
    ys = collect(-50:50) # target 1
    xs = collect(-50:50) # target 2

    cgs = Vector{CausalGraph}(undef, 100)
    for i=1:100
        cgs[i] = CausalGraph(SimpleDiGraph())

        # targets
        add_vertex!(cgs[i])
        set_prop!(cgs[i], 1, :object, Dot(pos = [0.0, ys[i], 0.0]))
        add_vertex!(cgs[i])
        set_prop!(cgs[i], 2, :object, Dot(pos = [xs[i], -20.0, 0.0]))
        
        # distractor
        add_vertex!(cgs[i])
        set_prop!(cgs[i], 3, :object, Dot(pos = [200, ys[i], 0.0]))
    end
    return cgs
end

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm_isr" => "$(@__DIR__)/dm_isr.json",
                 "dm_inertia" => "$(@__DIR__)/dm_inertia.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "k" => 100,
                 "viz" => true])
   
    # generating some data using the isr dynamics (using minimum distance)
    gm = MOT.load(GMParams, args["gm"])
    dm_inertia = MOT.load(InertiaModel, args["dm_inertia"])
    graphics = MOT.load(Graphics, args["graphics"])

    gt_cgs = get_split_cgs()[1:args["k"]]

    query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm,
                              dm_inertia,
                              graphics,
                              length(gt_cgs))
    
    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"],
                   objective = MOT.target_designation_receptive_fields)
                   
    # att = MOT.UniformAttention(sweeps = 2,
                               # ancestral_steps = 3)

    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    path = "/experiments/$(experiment_name)"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    
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
