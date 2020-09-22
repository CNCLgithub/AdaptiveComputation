using MOT
using CSV
using ArgParse
using FileIO
using DataFrames
using LinearAlgebra: norm

"""
    adds nearest distractor and distance
    to "attention_trial_tps.csv" 
"""
function add_nearest_distractor(att_tps::String, att_tps_out::String;
                                dataset_path::String="/datasets/exp0.jld2")
    df = DataFrame(CSV.File(att_tps))

    # adding new cols
    df[!,:nd] .= 0
    df[!,:dist_to_nd] .= 0.0
    df[!,:tracker_to_origin] .= 0.0 # perhaps to control for eccentricity?

    for (i, trial_row) in enumerate(eachrow(df))
        scene = trial_row.scene # indexing from R is 0-based
        scene_data = MOT.load_scene(scene, dataset_path, default_gm;
                                generate_masks=false)
        # getting the corresponding causal graph elements
        # (+1 because the first causal graph is for the init state)
        dots = scene_data[:gt_causal_graphs][trial_row.frame+1].elements
        pos = map(x->x.pos[1:2], dots)
        tracker_pos = pos[trial_row.tracker]

        df[i, :tracker_to_origin] = norm(tracker_pos - zeros(2))

        distances = map(distr_pos->norm(tracker_pos - distr_pos), pos[5:8])
        df[i, :nd] = argmin(distances)+4
        df[i, :dist_to_nd] = minimum(distances)
    end
    display(df)
    CSV.write(att_tps_out, df)
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

function render_probe_trial(trial_row::DataFrameRow, out::String;
                            pad::Int64 = 2, # how many frames from left and right of the peak
                            pad_end::Int64 = 8, # how many frames after the probe
                            probe::Bool = false)


    tracker, t, distractor = Tuple(trial_row[[:tracker, :frame, :nd]])
    trial_data = load_scene(q.scene, q.dataset_path, gm;
                            generate_masks=false)
    cgs = trial_data[:gt_causal_graphs]
    n_dots = round(Int, gm.n_trackers + gm.distractor_rate)
    if probe
        place_probes!(cgs, tracker, t, pad)
    end

    # rendering trial with tracker query
    tracker_out = "$(out)_trg"
    ispath(tracker_out) || mkpath(tracker_out)
    render(gm, min(t+pad+pad_end, q.k);
           gt_causal_graphs = cgs,
           path = tracker_out,
           stimuli = true,
           highlighted = [tracker],
           freeze_time = 24)

    # rendering trial with distractor query
    distractor_out = "$(out)_dis"
    ispath(distractor_out) || mkpath(distractor_out)
    render(gm, min(t+pad+pad_end, q.k);
           gt_causal_graphs = cgs,
           path = distractor_out,
           stimuli = true,
           highlighted = [distractor],
           freeze_time = 24)

    return nothing
end

function render_probe_trials(df, out::String,
                             max_t::Int64 = 120)
    for trow in eachrow(df)
        trial_out = joinpath(out, "$(trow.scene)_$(trow.tracker)_$(trow.epoch)")
        render_probe_trial(trow, "$(trial_out)_probe"; probe = true)
        render_probe_trial(trow, "$(trial_out)_noprobe"; probe = false)
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
    q = Exp0(scene = args["scene"])
    gm = MOT.load(GMMaskParams, q.gm)

    path = "/renders/probes"
    try
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    render_probe_trials(group, path)
end


main();
