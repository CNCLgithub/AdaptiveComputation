using MOT
using Random
using ImageTransformations:imresize
Random.seed!(1)

include("../analysis/attention_map.jl")
include("compile_videos.jl")

function get_probe_placement(exp_path::String, n_quantiles::Int; bin::Int=4)
    attmaps = load_attmaps(exp_path, bin=bin)
    b_ams = bool_attmaps(attmaps, n_quantiles)
    n_trials, n_quantiles = size(b_ams)
    probes = Array{Array{Bool}}(undef, size(b_ams))

    for i=1:n_trials
        for j=1:n_quantiles
            probes_trial_quant = zeros(Bool, size(b_ams[i,j]))
            b_ams[i,j][1:3,:] .= false
            b_ams[i,j][end-2:end,:] .= false
            idx = rand(1:count(b_ams[i,j]))
            coordinates = findall(x->x==true, b_ams[i,j])
            probes_trial_quant[coordinates[idx]] = true
            probes[i,j] = probes_trial_quant
        end
    end

    probes
end

function render_probes(q::AbstractExperiment,
                       exp_path::String,
                       out_path::String,
                       n_quantiles::Int;
                       bin::Int=4,
                       probe_duration::Int=4)

    probes = get_probe_placement(exp_path, n_quantiles, bin=bin)
    n_trials = size(probes,1)
    gm = MOT.load(GMMaskParams, q.gm)
    n_dots = round(Int, gm.n_trackers + gm.distractor_rate)

    for i=1:n_trials
        positions = last(load_trial(i, q.dataset_path, gm,
                                    generate_masks=false))
        for j=1:n_quantiles
            println("trial $i \t quantile $j")
            probes_trial_quant = zeros(Bool, q.k, n_dots)
            idx = first(findall(x->x==true, probes[i,j]))
            t = 1+(bin*(idx[1]-1))
            tracker = idx[2]
            probes_trial_quant[t:t+probe_duration-1, tracker] .= true
            
            path = joinpath(out_path, "$i", "$j")
            MOT.render(gm, dot_positions = positions, probes = probes_trial_quant, path = path,
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

main()
