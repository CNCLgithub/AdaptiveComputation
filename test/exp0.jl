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
exp = Exp0(;scene = args["trial"], k = args["time"],
            gm = args["gm"], proc = args["proc"])


function test_exp0(args, exp, att_mode)
    if att_mode == "target_designation"
        att = MOT.load(MapSensitivity, "/project/scripts/inference/exp0/td.json")
    elseif att_mode == "data_correspondence"
        att = MOT.load(MapSensitivity, "/project/scripts/inference/exp0/dc.json";
                    objective = MOT.data_correspondence)
    # else
    #     att = MOT.load(UniformAttention, ,
    #                 exp.scene, exp.k)
    end

    result_dir = "/project/test/exp0/$(get_name(exp))_$(att_mode)"
    path = joinpath(result_dir, "$(exp.scene)")
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
    run_inference(exp, att, out; viz = args["viz"])
end


test_exp0(args, exp, "target_designation")
test_exp0(args, exp, "data_correspondence")
