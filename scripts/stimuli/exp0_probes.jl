using DataFrames
using CSV
using FileIO
using VideoIO
using MOT

function place_probes(q::Exp0, tracker::T, t::T, pad::T) where {T<:Int}
    gm = MOT.load(GMMaskParams, q.gm)
    positions = last(load_exp0_trial(q.trial, gm, q.dataset_path))
    n_dots = round(Int, gm.n_trackers + gm.distractor_rate)
    probes = zeros(Bool, q.k, n_dots)
    t_end = min(q.k, t + pad)
    probes[t:t_end, tracker] .= true
    (gm, positions, probes)
end

function render_probe_trial(trial_row::DataFrameRow, out::String;
                            pad::Int64 = 4)

    trial = trial_row.scene + 1
    q = Exp0(k=120, trial = trial)

    args = Tuple(trial_row[[:tracker, :frame]])
    gm, pos, probes = place_probes(q, args..., pad)
    render(gm, dot_positions = pos, probes = probes, path = out,
           stimuli = true, highlighted = collect(1:4), freeze_time = 24)
    compile_video(out)

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

function render_probe_trials(att_tps::String)
    out = "/renders/probes"
    ispath(out) || mkpath(out)

    df = DataFrame(CSV.File(att_tps))
    display(df)
    for (i, trial_row) in enumerate(eachrow(df))
        trial = trial_row.scene
        trial_out = "$(out)/$i"
        ispath(trial_out) || mkpath(trial_out)
        render_probe_trial(trial_row, trial_out)
    end
end
