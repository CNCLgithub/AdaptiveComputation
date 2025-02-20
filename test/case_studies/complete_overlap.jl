using CSV
using MOT
using Accessors
using Gen_Compose
using UnicodePlots
using DataFrames

function get_states(gm, t)

    positions = []

    # right target that moves towards 0,0 from upper right
    pos2 = [80, 0]

    # distractor that gets closer to right tracker
    pos3 = [0, 0]

    push!(positions,
          [pos2, pos3])

    for k in 2:t
        # right target that moves towards 0,0 from upper right
        pos2 = positions[k-1][1] .+ [-3.0, 0.0]

        push!(positions, [pos2, pos3])
    end

    MOT.state_from_positions(gm, positions, [true, false])
end


experiment_name = "complete_overlap"

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "scene" => 1,
                 "chain" => 1,
                 "time" => 60,
                 "restart" => true,
                 "viz" => true,
                 "step_size" => 61])

    gm_params = InertiaGM(
        n_dots = 2,
        n_targets = 1,
        area_height = 800,
        area_width = 800,
        dot_radius = 20.0,
        vel = 1.0,
        bern = 0.98,
        k = 100.0,
        w = 0.5,
        wall_rep_m = 0.1,
        wall_rep_a = 0.1,
        wall_rep_x0 = 80.0,
        inner_f = 10.0,
        outer_f = 10.0,
        k_tail = 1,
        tail_sample_rate = 1,
    )
    gt_states = get_states(gm_params, args["time"])
    gt_init = first(gt_states)
    gt_states = gt_states[2:end]
    query = query_from_params(gm_params, gt_init, gt_states)

    att_mode = "target_designation"
    att = PopSensitivity(
        latents = 1,
        div_scale = 1.0,
        importance_tau = 1.0,
        init_samples = 10,
        min_samples = 5,
        max_arrousal = 35,
        x0 = 21.0,
        m = 0.40,
        plan = MOT.td_flat,
        plan_args = (20.0,),
        percept_update = tracker_kernel,
        percept_args = (3,)
    )
    proc = PopParticleFilter(
        particles = 20,
        attention = att,
    )

    base_path = "/spaths/test/$(experiment_name)_$(att_mode)"
    scene = args["scene"]
    path = joinpath(base_path, "$(scene)")

    try
        isdir(base_path) || mkpath(base_path)
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end

    df = DataFrame(
        chain = Int64[],
        t = Int64[],
        nabla = Float64[],
    )

    for ci = 1:5
        nsteps = length(gt_states) - 1
        logger = MemLogger(nsteps)
        chain = run_chain(proc, query, nsteps, logger)
        dg = extract_digest(logger)
        # visualize_inference(chain, dg, gt_states, gm_params, path)



        aux_state = dg[:, :auxillary]
        k = length(aux_state)
        task_relevance = Vector{Float64}(undef, k)
        for t=1:k
            task_relevance[t] = aux_state[t].sensitivities[1]
        end

        fig = scatterplot(1:k, task_relevance,
                        title="Task relevance")

        ylim = collect(extrema(filter(!isinf, task_relevance)))
        vline!(fig, 27, ylim)
        display(fig)

        append!(df, DataFrame(chain = ci, t = 1:k, nabla = task_relevance))
    end

    CSV.write("/spaths/test/$(experiment_name).csv", df)

    return nothing
end

results = main();
