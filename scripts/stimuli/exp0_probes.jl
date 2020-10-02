using MOT
using CSV
using ArgParse
using FileIO
using DataFrames
using LinearAlgebra: norm

"""
    rotates the positions of elements in causal graphs by angle in radians
"""
function rotate(cgs::Vector{CausalGraph}, angle::Float64)
    rotated_cgs = Vector{CausalGraph}(undef, length(cgs))

    for i=1:length(cgs)
        rotated_cgs[i] = deepcopy(cgs[i])
        for element in rotated_cgs[i].elements
            pos = element.pos[1:2]
            element.pos[1] = pos[1]*cos(angle) - pos[2]*sin(angle)
            element.pos[2] = pos[1]*sin(angle) + pos[2]*cos(angle)
        end
    end
    
    return rotated_cgs
end

# TODO: place padding to the left and righj of probe
# DONE?
function place_probes!(cgs, tracker::T, t::T, pad::T) where {T<:Int}
    t_start = max(1, t - pad)
    t_end = min(length(cgs), t + pad)
    for i = t_start:t_end
        dot = cgs[i+1].elements[tracker]
        cgs[i+1].elements[tracker] = Dot(pos = dot.pos,
                                       vel = dot.vel,
                                       probe = true,
                                       radius = dot.radius,
                                       width = dot.width,
                                       height = dot.height)
    end
end

function compile_movie(path::String)
    cmd = `ffmpeg -y -framerate 24 -i "$(path)/%03d.png"
   	    -hide_banner -crf 5 -preset slow
            -c:v libx264  -pix_fmt yuv420p "$(path).mp4"`
    run(cmd)
end

function render_probe_trial(q, gm, trial_row::DataFrameRow, out::String;
                            pad::Int64 = 1, # how many frames from left and right of the peak
                            pad_end::Int64 = 9, # how many frames after the probe
                            probe::Bool = false) # what do we query

    tracker, t, distractor = Tuple(trial_row[[:tracker, :frame, :nd]])
    scene_data = load_scene(q.scene, q.dataset_path, gm;
                            generate_masks=false)

    cgs = scene_data[:gt_causal_graphs]
    # making sure probed tracker is on top
    map(cg->cg.elements[tracker].pos[3] = -1.0, cgs)
    
    if probe
        cgs_probe = deepcopy(cgs)
        place_probes!(cgs_probe, tracker, t, pad)
    end

    t = probe ? min(t+pad+pad_end, q.k) : t # we add padding for the probe query

    # rendering trial with true query
    true_out = "$(out)_T"
    ispath(true_out) || mkpath(true_out)
    render(gm, t;
           gt_causal_graphs = probe ? cgs_probe : cgs,
           path = true_out,
           stimuli = true,
           highlighted = probe ? Int[] : [tracker],
           freeze_time = 24)

    # rendering trial with false query
    false_out = "$(out)_F"
    ispath(false_out) || mkpath(false_out)
    render(gm, t;
           gt_causal_graphs = cgs,
           path = false_out,
           stimuli = true,
           highlighted = probe ? Int[] : [distractor],
           freeze_time = 24)

    compile_movie(true_out)
    compile_movie(false_out)

    return nothing
end

function render_probe_trials(q, gm, df, out::String)
    for trow in eachrow(df)
        trial_out = joinpath(out, "$(trow.scene)_$(trow.tracker)_$(trow.epoch)")
        render_probe_trial(q, gm, trow, "$(trial_out)_pr"; probe = true)
        render_probe_trial(q, gm, trow, "$(trial_out)_td"; probe = false)
    end
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin

        "probe_map"
        help = "Probe map"
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
    df = DataFrame(CSV.File(args["probe_map"]))
    scenes = groupby(df, :scene)
    scene = keys(scenes)[args["scene"]]
    group = scenes[scene]
    q = Exp0(scene = scene[1]) # was bug here
    gm = MOT.load(GMMaskParams, q.gm)
   
    # filtering out epochs 2 and 4
    filter!(row->row[:epoch] in ["t_1", "t_3", "t_5"], df)

    path = "/renders/probes"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    render_probe_trials(q, gm, group, path)
end


main();
