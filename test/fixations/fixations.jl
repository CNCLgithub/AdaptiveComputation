using CSV
using MOT
using ArgParse
using Setfield
using Profile
using StatProfilerHTML
using Lazy: @>>
using Statistics: norm, mean, std, var

"""
    returns stats
"""
function get_stats(cgs::Vector{MOT.CausalGraph})::NamedTuple
    n_frames = length(cgs)
    n_objects = length(MOT.get_objects(first(cgs), Dot))

    pos = Array{Float64}(undef, n_frames, n_objects, 2)
    for i=1:n_frames
        dots = MOT.get_objects(cgs[i], Dot)
        for j=1:n_objects
            pos[i,j,:] = dots[j].pos[1:2]
        end
    end

    pos_t0 = pos[1:end-1,:,:]
    pos_t1 = pos[2:end,:,:]
    delta_pos = pos_t1 - pos_t0
    
    # getting velocity vectors
    vels = @>> Iterators.product(1:n_frames-1, 1:n_objects) begin
        map(ij -> delta_pos[ij[1], ij[2], :])
    end
    
    # magnitude
    mags = norm.(vels)
    vel_mu = mean(mags)
    vel_std = std(mags, mean=vel_mu)
    
    # angle
    angs = @>> vels begin
        map(vel -> atan(vel...))
    end
    angs_t0 = angs[1:end-1,:,:]
    angs_t1 = angs[2:end,:,:]
    delta_angs = angs_t1 - angs_t0
    
    # when kappa is large (small variance), it's approximately a normal distribution:
    # https://en.wikipedia.org/wiki/Von_Mises_distribution#Limiting_behavior
    ang_var = var(delta_angs)
    ang_kappa = 1/ang_var

    stats = (vel_mu = vel_mu,
             vel_std = vel_std,
             ang_var = ang_var,
             ang_kappa = ang_kappa)
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gm"
        help = "Generative Model params"
        arg_type = String
        default = "$(@__DIR__)/gm.json"

        "--proc"
        help = "Inference procedure params"
        arg_type = String
        default = "$(@__DIR__)/proc.json"

        "--dm"
        help = "Motion parameters for Inertia model"
        arg_type = String
        default = "$(@__DIR__)/dm.json"

        "--graphics"
        help = "Receptive fields parameters"
        arg_type = String
        default = "$(@__DIR__)/graphics.json"

        "--dataset"
        help = "jld2 dataset path"
        arg_type = String
        default = joinpath("/datasets", "fixations_dataset.jld2")

        "--fps", "-f"
        help = "Frames per second"
        arg_type = Int64
        default = 24

        "--fpsdataset", "-f"
        help = "Frames per second dataset"
        arg_type = Int64
        default = 60

        "--time", "-t"
        help = "Seconds to track"
        arg_type = Int64
        default = 10

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "--viz", "-v"
        help = "Whether to render masks"
        action = :store_true

        "scene"
        help = "Which scene to run"
        arg_type = Int
        required = true

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        required = true

        "target_designation", "T"
        help = "Using target designation"
        action = :command

        "data_correspondence", "D"
        help = "Using data correspondence"
        action = :command

        "scene_avg", "A"
        help = "Using scene avg"
        action = :command

    end

    @add_arg_table! s["target_designation"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/td.json"

    end
    @add_arg_table! s["data_correspondence"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/dc.json"
    end
    @add_arg_table! s["scene_avg"] begin
        "model_path"
        help = "path containing compute allocations"
        arg_type = String
        required = true

    end

    return parse_args(s)
end

experiment_name = "fixations"

function main()
    #args = parse_commandline()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm" => "$(@__DIR__)/dm.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "dataset" => "/datasets/fixations_dataset.jld2",
                 "scene" => 10,
                 "chain" => 1,
                 "fps" => 60,
                 "fpsdataset" => 60,
                 "time" => 1.0, # this is now seconds
                 "restart" => true,
                 "viz" => true])

    scene_data = load_scene(args["scene"], args["dataset"])
    frames_per_step = round(Int64, args["fpsdataset"] / args["fps"])
    last_frame = round(Int64, args["time"] * args["fpsdataset"])

    @show frames_per_step
    @show last_frame

    gt_cgs = scene_data[:gt_causal_graphs][1:frames_per_step:last_frame]
    aux_data = scene_data[:aux_data]
    scene_stats = get_stats(gt_cgs) 

    @show aux_data
    @show scene_stats

    gm_params = MOT.load(GMParams, args["gm"])
    gm_params = @set gm_params.n_trackers = sum(aux_data[:targets])
    gm_params = @set gm_params.distractor_rate = count(iszero, aux_data[:targets])

    dm_params = MOT.load(InertiaModel, args["dm"])
    dm_params = @set dm_params.vel = scene_stats.vel_mu
    dm_params = @set dm_params.k_max = scene_stats.ang_kappa

    graphics_params = MOT.load(Graphics, args["graphics"])

    prof_query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm_params,
                              dm_params,
                              graphics_params,
                              1)
    query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm_params,
                              dm_params,
                              graphics_params,
                              length(gt_cgs))

    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"],
                   objective = MOT.target_designation_receptive_fields,
                   weights = fill(-50.0, sum(aux_data[:targets])))

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
    Profile.init(delay = 0.001,
                 n = 10^6)
    # @profilehtml results = run_inference(prof_query, proc)
    # @profilehtml results = run_inference(query, proc)
    results = run_inference(query, proc)

    df = MOT.analyze_chain_receptive_fields(results,
                                            n_trackers = gm_params.n_trackers,
                                            n_dots = gm_params.n_trackers + gm_params.distractor_rate,
                                            gt_cg_end = gt_cgs[end])
    df[!, :scene] .= args["scene"]
    df[!, :chain] .= c
    CSV.write(joinpath(path, "$(c).csv"), df)

    if (args["viz"])
        visualize_inference(results, gt_cgs, gm_params,
                            graphics_params, att, path)
    end

    return results
end

results = main();
