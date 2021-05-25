using CSV
using Gen
using MOT
using MOT: CausalGraph
using LightGraphs: SimpleDiGraph, add_vertex!
using MetaGraphs: set_prop!, get_prop
using ArgParse
using Setfield
using Profile
using StatProfilerHTML
using Lazy: @>, @>>
using Statistics: norm, mean, std, var

function mh_move(trace, tracker, steps)
    t = first(Gen.get_args(trace))
    addrs = []
    for i = max(1, t-steps):t
        addr = :kernel => i => :dynamics => :brownian => tracker
        push!(addrs, addr)
    end
    first(regenerate(trace, Gen.select(addrs...)))
end

function display_trace(trace, t)
    _, cgs = Gen.get_retval(trace)
    cg = cgs[t]
    @> cg begin
        MOT.get_objects(MOT.Dot)
        (@>> map(MOT.get_pos))
        collect
        display
    end
end


function main()
    args = Dict(["target_designation" => Dict(["params" => "$(@__DIR__)/td.json"]),
                 "dm" => "$(@__DIR__)/dm.json",
                 "graphics" => "$(@__DIR__)/graphics.json",
                 "gm" => "$(@__DIR__)/gm.json",
                 "proc" => "$(@__DIR__)/proc.json",
                 "dataset" => "/datasets/fixations_dataset.jld2",
                 "scene" => 10,
                 "chain" => 1,
                 "fps" => 60,
                 "frames" => 100,
                 "restart" => true,
                 "viz" => true])


    gm_params = MOT.load(GMParams, args["gm"])
    gm_params = @set gm_params.n_trackers = 2
    gm_params = @set gm_params.distractor_rate = 1

    dm_params = MOT.load(InertiaModel, args["dm"])
    graphics_params = MOT.load(Graphics, args["graphics"])

    (trace, ls) = Gen.generate(gm_inertia_mask,
                               (2, gm_params, dm_params, graphics_params))

    @show "trace a"
    display_trace(trace, 1)
    tr = mh_move(trace, 1, 1)
    @show "trace a after create trace b"
    display_trace(trace, 1)
    @show "trace b"
    display_trace(tr, 1)
    return nothing
end

main();
