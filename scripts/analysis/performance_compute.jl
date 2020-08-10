using MOT

using HDF5
using Gadfly
using Statistics
using DataFrames
using CSV

function performance_compute(src::String)
    results = load_results(src)
    perf_trial = mean(results["performance"], dims=2)[:,1]
    comp_trial = mean(results["compute"], dims=2)[:,1]
    pred_target_trial = mean(results["pred_target"], dims=2)[:,1,:]

    pred_target_packaged = []
    for i=1:size(pred_target_trial, 1)
        push!(pred_target_packaged, pred_target_trial[i,:])
    end

    trials = collect(1:length(perf_trial))

    df = DataFrame(performance=perf_trial, compute=comp_trial,
                    trial=trials;
                    [Symbol("dot_$i")=>pred_target_trial[:,i] for i=1:8]...
                    )

    CSV.write("$(src)/performance_compute.csv", df)
    println(df)
end
