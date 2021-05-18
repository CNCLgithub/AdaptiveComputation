using JSON
using JLD2
using FileIO
using Lazy: @>>


include("helpers_constants.jl")

"""
    goes through the subject's data and adds
    to the trial_fixations matrix accordingly
"""
function get_trial_fixations!(trial_fixations, sub_data)
    for frame_data in sub_data
        i = frame_data["Trial number"]
        j = frame_data["Frame number"]
        k = frame_data["Subject ID"] + 1

        x = frame_data["fixation.x"]
        y = frame_data["fixation.y"]
        
        trial_fixations[i, j, k, :] = translate(x, y)
    end
end

dir = joinpath("output", "fixations", "MOT_json_files")
fn_start = "behavior test trajectories_0_0_random_with fixation from sub_"
fns = @>> readdir(dir) filter(fn -> occursin(fn_start, fn))

# huge but still fine
data = @>> fns map(fn -> JSON.parsefile(joinpath(dir, fn)))

n_trials = 120
n_frames = 600
n_subjects = 50

trial_fixations = fill(NaN, n_trials, n_frames, n_subjects, 2)
for (i, sub_data) in enumerate(data)
    print("going through subject's $i data \r")
    get_trial_fixations!(trial_fixations, sub_data)
end

@show trial_fixations

outdir = joinpath("output", "fixations", "parsed_fixations")
outpath = joinpath(outdir, "trial_fixations.jld2")
save(outpath, Dict("trial_fixations" => trial_fixations,
                   "dimensions" => "n_trials x n_frames x n_subjects x 2"))


