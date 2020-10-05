using MOT

args = Dict(
    "trial" => 51,
    "chain" => 1,
    "time" => 5,
    "gm" => "/project/scripts/inference/exp0/gm.json",
    "proc" => "/project/scripts/inference/exp0/proc.json",
    "restart" => true,
    "viz" => true
)

function test_exp0(args, att_mode)
    if att_mode == "target_designation"
        att = MOT.load(MapSensitivity, "/project/scripts/inference/exp0/td.json")
    elseif att_mode == "data_correspondence"
        att = MOT.load(MapSensitivity, "/project/scripts/inference/exp0/dc.json";
                    objective = MOT.data_correspondence)
    end


    query = query_from_params(args["gm"], args["dataset"],
                          args["trial"], args["time"])

    proc = load(PopParticleFilter, args["proc"];
                rejuvenation = rejuvenate_attention!,
                rejuv_args = (att,))

    result_dir = "/project/test/exp0/brownian_$(att_mode)"
    scene = args["trial"]
    path = joinpath(result_dir, "$(scene)")
    try
        isdir(result_dir) || mkpath(result_dir)
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
    run_inference(query, proc, out; viz = args["viz"])
end


test_exp0(args, exp, "target_designation")
test_exp0(args, exp, "data_correspondence")
