using Base.Filesystem

using HDF5
using ArgParse
using MOT

function render(dataset, idx, n = 220)
    df_path = joinpath("/datasets", "$(dataset).h5")
    scene_path = joinpath("/dataset", "$(idx)")

    obs = h5read(df_path, joinpath(scene_path, "obs"))
    n = h5read(df_path, joinpath(scene_path, "num_observations"))
    nt = h5read(df_path, joinpath(scene_path, "num_targets"))


    for k = 1:n
        outpath = joinpath("/renders", dataset, "$(idx)_$(k)")
        mkpath(outpath)
        if length(readdir(outpath)) >= n
            println("Already completed $(outpath)")
            continue
        end
        # render images
        overlay(obs, nt, outpath,
                stimuli = true,
                highlighted = [k],
                colors = ["blue"])
    end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "dataset"
        help = "dataset to load scene info"
        arg_type = String
        required = true

        "scene"
        help = "dataset to load scene info"
        arg_type = Int
        required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    render(args["dataset"], args["scene"])
end

main()
