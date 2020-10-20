using MOT
using Random

Random.seed!(1)

include("attention_map.jl")

function get_difficulty(c::Dict, bin::Int)
    attmap = attention_map(c, bin)
    difficulty = sum(attmap, dims=2)
end

function load_difficulty(trial_path::String; bin::Int = 4)
    attmap = load_attmap(trial_path, bin=bin)
    difficulty = sum(attmap, dims=2)
end

"""
    z scored difficulty maps from an experiment results folder
"""
function load_difficulties(exp_path::String; bin::Int64 = 4, z_scored=true)
    trials = readdir(exp_path)
    trials = filter(isdir, joinpath.(exp_path, trials))
    difficulty = load_difficulty(joinpath(exp_path, first(trials)), bin=bin)
    difficulties = Array{Float64}(undef, length(trials), size(difficulty)...)
    for (i, trial) in enumerate(trials)
        print("getting difficulty for trial $i \r")
        difficulties[i,:] = load_difficulty(joinpath(exp_path, trial))
    end
    print("done                               \n")
    if z_scored
        mu = mean(difficulties)
        std = Statistics.std(difficulties)
        difficulties = (difficulties .- mu)/std
    end
    difficulties
end

"""
    returns boolean difficulty maps for when to stop the trial
    n - number of difficulty levels (quantiles)
"""
function bool_difficulties(difficulties::Vector{Array{Float64}}, n_quantiles::Int)
    # b_diffs = bool_attmaps(difficulties, n_quantiles)
end

