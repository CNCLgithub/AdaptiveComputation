using DataFrames
using CSV
using FileIO
using MOT

function place_probes(q::Exp0, tracker::T, t::T, pad::T) where {T<:Int}
    gm = MOT.load(GMMaskParams, q.gm)
    positions = last(load_exp0_trial(q.trial, gm, q.dataset_path))
    n_dots = round(Int, gm.n_trackers + gm.distractor_rate)
    probes = zeros(Bool, q.k, n_dots)
    t_end = min(q.k, t + pad)
    probes[t:t_end, 4] .= true
    (gm, positions, probes)
end

function render_probe_trial(trial_row::DataFrameRow, out::String;
                            pad::Int64 = 4)

    trial = trial_row.scene + 1
    q = Exp0(k=120, trial = trial)

    td_out = joinpath(out, "td")
    gm, pos, probes = place_probes(q, Tuple(trial_row[[:td_tracker, :td_t]])..., pad)
    render(gm, dot_positions = pos, probes = probes, path = td_out,
           stimuli = false, highlighted = collect(1:4))

    dc_out = joinpath(out, "dc")
    gm, pos, probes = place_probes(q, Tuple(trial_row[[:dc_tracker, :dc_t]])..., pad)
    render(gm, dot_positions = pos, probes = probes, path = dc_out,
           stimuli = false, highlighted = collect(1:4))
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
