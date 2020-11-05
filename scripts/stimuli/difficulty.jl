using MOT
using CSV
using ArgParse
using FileIO
using DataFrames
using LinearAlgebra: norm


function compile_movie(path::String)
    cmd = `ffmpeg -y -framerate 24 -i "$(path)/%03d.png"
   	    -hide_banner -crf 5 -preset slow
            -c:v libx264  -pix_fmt yuv420p "$(path).mp4"`
    run(cmd)
end

function render_diff_trial(scene_data, trial_row::DataFrameRow, out::String)
    t = trial_row[:frame]
    cgs = deepcopy(scene_data[:gt_causal_graphs])
    
    for tracker=1:8
        tracker_out = "$(out)_$(tracker)"
        ispath(tracker_out) || mkpath(tracker_out)

        render(default_gm, t;
               gt_causal_graphs = cgs,
               path = tracker_out,
               stimuli = true,
               highlighted = [tracker],
               freeze_time = 24)

        compile_movie(tracker_out)
    end


    return nothing
end

function render_diff_trials(scene_data, df, out::String)
    println(df)
    for trow in eachrow(df)
        trial_out = joinpath(out, "$(trow.scene)_t_$(trow.epoch)")
        render_diff_trial(scene_data, trow, "$(trial_out)")
    end
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin

        "diff_map"
        help = "Probe map"
        arg_type = String
        required = true

        "dataset_path"
        help = "Dataset for scenes"
        arg_type = String
        required = true

        "scene"
        help = "Which scene to render"
        arg_type = Int
        required = true

    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    # args = Dict("scene" => 2,
                # "diff_map" => "output/attention_analysis/exp0_diff_map_nd.csv")

    df = DataFrame(CSV.File(args["diff_map"]))
    scenes = groupby(df, :scene)
    scene = keys(scenes)[args["scene"]]
    group = scenes[scene]

    scene_data = MOT.load_scene(scene[1], args["dataset_path"], default_gm;
                                generate_masks=false)
   
    path = "/renders/difficulty"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    render_diff_trials(scene_data, group, path)
end


main();
