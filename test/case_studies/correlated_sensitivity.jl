using CSV
using MOT
using Accessors
using Gen_Compose



function get_states(gm, t)

    positions = []

    # left target that moves along x = -200
    pos1 = [-200.0, 310]


    # right target that moves towards 0,0 from upper right
    pos2 = [190.0, 60]

    # distractor that gets closer to right tracker
    pos3 = [200.0, 0]

    push!(positions,
          [pos1, pos2, pos3])

    mag = 5.0
    ang = (pi / 3.0)
    vx = mag * cos(ang)
    vy = mag * sin(ang)

    for k in 2:t
        # left target that moves along x = -200
        pos1 = positions[k-1][1] .+ [0, -mag]

        # right target that moves towards 0,0 from upper right
        pos2 = positions[k-1][2] .+ [-vx, -vy]

        # distractor that gets closer to right tracker
        pos3 = positions[k-1][3] .+ [-mag, 0.0]

        push!(positions, [pos1, pos2, pos3])
    end

    MOT.state_from_positions(gm, positions, [true, true, false])
end


experiment_name = "correlated_sensitivity"

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "scene" => 1,
                 "chain" => 1,
                 "time" => 81,
                 "restart" => true,
                 "viz" => true,
                 "step_size" => 61])

    gm_params = MOT.load(InertiaGM, args["gm"])
    gt_states = get_states(gm_params, args["time"])
    gt_init = first(gt_states)
    gt_states = gt_states[2:end]
    query = query_from_params(gm_params, gt_init, gt_states)

    att_mode = "target_designation"
    att = MOT.load(PopSensitivity, args[att_mode]["params"],
                   plan = MOT.td_flat,
                   plan_args = (10.0,),
                   percept_update = tracker_kernel,
                   percept_args = (3,)
                   )

    proc = MOT.load(PopParticleFilter, args["proc"];
                    attention = att)

    base_path = "/spaths/experiments/$(experiment_name)_$(att_mode)"
    scene = args["scene"]
    path = joinpath(base_path, "$(scene)")
    try
        isdir(base_path) || mkpath(base_path)
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    nsteps = length(gt_states) - 1
    logger = MemLogger(nsteps)
    chain = run_chain(proc, query, nsteps, logger)
    dg = extract_digest(logger)
    visualize_inference(chain, dg, gt_states, gm_params, path)

    return nothing
end

results = main();
