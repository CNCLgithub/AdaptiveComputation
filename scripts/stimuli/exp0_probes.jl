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

    td_out = joinpath(out, "td")
    td_args = Tuple(trial_row[[:td_tracker, :td_t]])
    gm, pos, probes = place_probes(q, td_args..., pad)
    render(gm, dot_positions = pos, probes = probes, path = td_out,
           stimuli = true, highlighted = [1], freeze_time = 24)
    compile_video(out, "td")

    dc_out = joinpath(out, "dc")
    dc_args = Tuple(trial_row[[:dc_tracker, :dc_t]])
    _, _, probes = place_probes(q, dc_args..., pad)
    render(gm, dot_positions = pos, probes = probes, path = dc_out,
           stimuli = true, highlighted = [1], freeze_time = 24)
    compile_video(out, "dc")
    display(td_args)
    display(dc_args)
    return nothing
end

function compile_video(path::String, model::String)
    render_path = joinpath(path, model)
    imgnames = filter(x->occursin(".png",x), readdir(render_path))
    intstrings =  map(x->split(x,".")[1],imgnames)
    p = sortperm(parse.(Int,intstrings))
    imgstack = []
    for imgname in imgnames[p]
        push!(imgstack,load(joinpath(render_path, imgname)))
    end
    encodevideo("$(path)_$(model).mp4", imgstack)
end

function render_probe_trials(att_tps::String)
    out = "/renders/probes"
    ispath(out) || mkpath(out)

    df = DataFrame(CSV.File(att_tps))
    for trial_row in eachrow(df)
        trial = trial_row.scene
        trial_out = "$(out)/$(trial)"
        ispath(trial_out) || mkpath(trial_out)
        render_probe_trial(trial_row, trial_out)
    end
end
