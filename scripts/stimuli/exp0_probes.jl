using DataFrames
using CSV
using FileIO
using MOT
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
    df[!,:tracker_pos_x] .= 0.0 # perhaps to control for eccentricity?
    df[!,:tracker_pos_y] .= 0.0 # --||--

    for (i, trial_row) in enumerate(eachrow(df))
        scene = trial_row.scene
        scene_data = load_scene(scene, dataset_path, default_gm;
                                generate_masks=false)
        # getting the corresponding causal graph elements
        # (+1 because the first causal graph is for the init state)
        dots = scene_data[:gt_causal_graphs][trial_row.frame+1].elements
        pos = map(x->x.pos[1:2], dots)
        tracker_pos = pos[trial_row.tracker]

        df[i, :tracker_pos_x] = tracker_pos[1]
        df[i, :tracker_pos_y] = tracker_pos[2]

        distances = map(distr_pos->norm(tracker_pos - distr_pos), pos[5:8])
        display(distances)
        df[i, :nd] = argmin(distances)+4
        df[i, :dist_to_nd] = minimum(distances)
    end
    CSV.write(att_tps_out, df)
end


# TODO: place padding to the left and righj of probe
# DONE?
function place_probes!(cgs, tracker::T, t::T, pad::T) where {T<:Int}
    t_start = max(1, t - pad)
    t_end = min(length(cgs), t + pad)
    for i = t_start:t_end
        dot = cgs[i].elements[tracker]
        cgs[i].elements[tracker] = Dot(pos = dot.pos,
                                       vel = dot.vel,
                                       probe = true,
                                       radius = dot.radius,
                                       width = dot.width,
                                       height = dot.height)
    end
end

function render_probe_trial(trial_row::DataFrameRow, out::String;
                            pad::Int64 = 2,
                            probe::Bool = false)

    trial = trial_row.scene + 1
    q = Exp0(k=120, trial = trial)

    tracker, t = Tuple(trial_row[[:tracker, :frame]])
    gm = MOT.load(GMMaskParams, q.gm)
    trial_data = load_trial(q.trial, q.dataset_path, gm)
    cgs = trial_data[:gt_causal_graphs]
    n_dots = round(Int, gm.n_trackers + gm.distractor_rate)
    if probe
        place_probes!(cgs, tracker, t, pad)
    end
    render(gm, q.k;
           gt_causal_graphs = cgs,
           path = out,
           stimuli = true, highlighted = collect(1:4), freeze_time = 24)

    return nothing
end

function render_probe_trials(att_tps::String; pct_control::Float64 = 0.5)
    out = "/renders/probes"
    ispath(out) || mkpath(out)
    df = DataFrame(CSV.File(att_tps))
    max_probes = Int64((1.0-pct_control) * nrow(df))
    display(df)
    for (i, trial_row) in enumerate(eachrow(df))
        trial = trial_row.scene
        trial_out = "$(out)/$i"
        ispath(trial_out) || mkpath(trial_out)
        render_probe_trial(trial_row, trial_out; probe = i <= max_probes)
    end
end
