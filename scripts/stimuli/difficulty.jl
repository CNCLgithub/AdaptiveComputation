using MOT
using Random
using ImageTransformations:imresize
Random.seed!(1)

include("../analysis/difficulty_map.jl")
include("compile_videos.jl")

function get_difficulty_times(exp_path::String, n_quantiles::Int; bin::Int=4)
    difficulties = load_difficulties(exp_path, bin=bin, z_scored=false) # CHANGE remove z_scored=false
    diffs = Vector{Array{Float64}}(undef, size(difficulties,1))
    for i=1:size(difficulties,1)
        diffs[i] = difficulties[i,:]
    end
    display(difficulties)
    display(diffs)
    b_diffs = bool_difficulties(diffs, n_quantiles)
    n_trials, n_quantiles = size(b_diffs)
    diff_times = Array{Int}(undef, n_trials, n_quantiles)

    for i=1:n_trials
        for j=1:n_quantiles
            b_diffs[i,j][1:3,:] .= false
            b_diffs[i,j][end-2:end,:] .= false
            idx = rand(1:count(b_diffs[i,j]))
            coordinates = findall(x->x==true, b_diffs[i,j])
            diff_times[i,j] = coordinates[idx]
        end
    end
    
    diff_times
end

function render_difficulty(q::AbstractExperiment,
                           exp_path::String,
                           out_path::String,
                           n_quantiles::Int;
                           bin::Int=4)

    diff_times = get_difficulty_times(exp_path, n_quantiles, bin=bin)
    n_trials = size(diff_times,1)
    gm = MOT.load(GMMaskParams, q.gm)
    n_dots = round(Int, gm.n_trackers + gm.distractor_rate)
    
    println("hello")
    display(diff_times)

    for i=1:n_trials
        positions = last(load_trial(i, q.dataset_path, gm,
                                    generate_masks=false))
        for j=1:n_quantiles
            println("trial $i \t quantile $j")
            # TODO make this proper (find time in non-binned time)
            time = round(Int, 1+(bin*(diff_times[i,j]-1)) + bin/2)
            
            path = joinpath(out_path, "$i", "$j")
            MOT.render(gm, dot_positions = positions[1:time], path = path,
                       stimuli = true, highlighted = collect(1:4), freeze_time = 24)
        end
    end
end

function main()
    q = Exp1(gm="scripts/inference/exp1/gm.json",
             dataset_path="/datasets/exp1_isr.jld2")
    n_quantiles = 5
    
    println("rendering probes for target_designation")
    exp_path = "/experiments/exp1_isr/exp1_target_designation" # CHANGE
    out_path = "/renders/exp1_isr/exp1_target_designation"
    render_probes(q, exp_path, out_path, n_quantiles)
    videos_out_path = "/renders/videos/exp1_isr/exp1_target_designation"
    compile_videos(out_path, videos_out_path)

    println("rendering probes for data_correspondence")
    exp_path = "/experiments/exp1_isr/exp1_data_correspondence"
    out_path = "/renders/exp1_isr/exp1_data_correspondence"
    render_probes(q, exp_path, out_path, n_quantiles)
    videos_out_path = "/renders/videos/exp1_isr/exp1_data_correspondence"
    compile_videos(out_path, videos_out_path)
end

# main()
