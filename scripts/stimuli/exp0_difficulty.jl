using DataFrames
using CSV
using FileIO
using VideoIO
using MOT
using LinearAlgebra: norm


function render_difficulty_trial(trial_row::DataFrameRow, out::String;
                                 difficulty::Bool = false)

    q = Exp0(scene = trial_row.scene)
    
    # stopping either at the frame determined by
    # attention or the end of the trial
    t = difficulty ? trial_row[:frame] : q.k

    gm = MOT.load(GMMaskParams, q.gm)
    scene_data = load_scene(q.scene, q.dataset_path, gm;
                            generate_masks=false)
    cgs = scene_data[:gt_causal_graphs]
    
    # let's just do first tracker, first distractors?
    tracker = 1
    distractor = 5

    # rendering trial with tracker query
    tracker_out = "$(out)_1"
    ispath(tracker_out) || mkpath(tracker_out)
    render(gm, t;
           gt_causal_graphs = cgs,
           path = tracker_out,
           stimuli = true,
           highlighted = [tracker],
           freeze_time = 24)

    # rendering trial with distractor query
    distractor_out = "$(out)_2"
    ispath(distractor_out) || mkpath(distractor_out)
    render(gm, t;
           gt_causal_graphs = cgs,
           path = distractor_out,
           stimuli = true,
           highlighted = [distractor],
           freeze_time = 24)

    return nothing
end

function render_difficulty_trials(diff_tps::String; pct_control::Float64 = 0.5)
    out = "/renders/difficulty"
    ispath(out) || mkpath(out)
    df = DataFrame(CSV.File(diff_tps))
    max_difficulty = Int64((1.0-pct_control) * nrow(df))
    display(df)
    for (i, trial_row) in enumerate(eachrow(df))
        trial_out = "$(out)/$i"
        render_difficulty_trial(trial_row, trial_out;
                                difficulty = i <= max_difficulty)
    end
end
