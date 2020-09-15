using DataFrames
using CSV
using FileIO
using VideoIO
using MOT

function place_probes!(cgs, tracker::T, t::T, pad::T) where {T<:Int}
    t_end = min(length(cgs), t + pad)
    for i = t:t_end
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
                            pad::Int64 = 4,
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
    # compile_video(out)

    return nothing
end

function compile_video(render_path::String)
    imgnames = filter(x->occursin(".png",x), readdir(render_path))
    intstrings =  map(x->split(x,".")[1],imgnames)
    p = sortperm(parse.(Int,intstrings))
    imgstack = []
    for imgname in imgnames[p]
        push!(imgstack,load(joinpath(render_path, imgname)))
    end
    encodevideo("$(render_path).mp4", imgstack)
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
