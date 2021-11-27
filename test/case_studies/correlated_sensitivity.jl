using Gen_Compose
using CSV
using MOT
using MOT: CausalGraph
using LightGraphs: SimpleDiGraph, add_vertex!
using MetaGraphs
using MetaGraphs: set_prop!, get_prop
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


function get_cgs(t::Int64)

    cgs = Vector{CausalGraph}(undef, t)


    cg = CausalGraph(SimpleDiGraph())
    # left target that moves along x = -200
    add_vertex!(cg)
    v = MetaGraphs.nv(cg)
    set_prop!(cg, v, :object, Dot(pos = [-200.0, 210, 0],
                                  target = 1.0))

    # right target that moves towards 0,0 from upper right
    add_vertex!(cg)
    v = MetaGraphs.nv(cg)
    set_prop!(cg, v, :object, Dot(pos = [190.0, 60, 0],
                                  target = 1.0))

    # distractor that gets closer to right tracker
    add_vertex!(cg)
    v = MetaGraphs.nv(cg)
    set_prop!(cg, v, :object, Dot(pos = [200.0, 0, 0]))

    cgs[1] = cg

    for k in 2:t
        cg = CausalGraph(SimpleDiGraph())
        # left target that moves along x = -200
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        dot = get_prop(cgs[k-1], v, :object)
        dot = @set dot.pos = dot.pos + [0, -2.0, 0]
        set_prop!(cg, v, :object, dot)

        # right target that moves towards 0,0 from upper right
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        dot = get_prop(cgs[k-1], v, :object)
        mag = 2.0
        ang = (pi / 3.0)
        vx = mag * cos(ang)
        vy = mag * sin(ang)
        dot = @set dot.pos = dot.pos + [-vx, -vy, 0]
        set_prop!(cg, v, :object, dot)

        # distractor that gets closer to right tracker
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        dot = get_prop(cgs[k-1], v, :object)
        dot = @set dot.pos = dot.pos + [-2.0, 0.0, 0]
        set_prop!(cg, v, :object, dot)

        cgs[k] = cg
    end

    scene = Dict{Symbol, Any}(
    :gt_causal_graphs => cgs,
    :targets => [1,1,0])
    return scene
end


experiment_name = "correlated_sensitivity"

function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm" => "$(@__DIR__)/dm.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "scene" => 10,
                 "chain" => 1,
                 "fps" => 60,
                 "frames" => 31,
                 "restart" => true,
                 "viz" => true,
                 "step_size" => 10])


    scene_data = get_cgs(args["frames"])

    gt_cgs = scene_data[:gt_causal_graphs]
    # gt_cgs = scene_data[:gt_causal_graphs][45:65]
    scene_stats = get_stats(gt_cgs)

    @show scene_stats

    gm_params = MOT.load(GMParams, args["gm"])
    gm_params = @set gm_params.n_targets = sum(scene_data[:targets])
    gm_params = @set gm_params.max_things = gm_params.n_targets + count(iszero, scene_data[:targets])

    dm_params = MOT.load(InertiaModel, args["dm"])
    dm_params = @set dm_params.vel = scene_stats.vel_mu
    # dm_params = @set dm_params.k_max = scene_stats.ang_kappa

    graphics_params = MOT.load(Graphics, args["graphics"])

    query = query_from_params(gt_cgs,
                              gm_inertia_mask,
                              gm_inertia_mask,
                              gm_params,
                              dm_params,
                              graphics_params,
                              length(gt_cgs))

    att_mode = "target_designation"
    att = MOT.load(MapSensitivity, args[att_mode]["params"],
                   objective = MOT.td_flat,
                   )

    proc = MOT.load(PopParticleFilter, args["proc"];
                    rejuvenation = rejuvenate_attention!,
                    rejuv_args = (att,))

    base_path = "/spaths/experiments/$(experiment_name)_$(att_mode)"
    scene = args["scene"]
    path = joinpath(base_path, "$(scene)")
    try
        isdir(base_path) || mkpath(base_path)
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    c = args["chain"]
    chain_path = joinpath(path, "$(c).jld2")

    println("running chain $c")
    isfile(chain_path) && rm(chain_path)
    chain = sequential_monte_carlo(proc, query, chain_path,
                                    args["step_size"])
    if (args["viz"])
        visualize_inference(chain, chain_path, gt_cgs, gm_params,
                            graphics_params, att, path)
    end

    return nothing
end

results = main();
