using MOT
using CSV
using Statistics
using LinearAlgebra:norm
using Base.Iterators:take
using DataFrames
using Gadfly
Gadfly.push_theme(Theme(background_color = colorant"white"))
using Compose
import Cairo
using ImageTransformations:imresize
using Statistics
using StatsBase

function attention_map(c::Dict, bin::Int64)
    aux_state = c["aux_state"]
    k = length(aux_state)
    n = length(first(aux_state).attended_trackers)
    attended = Matrix{Float64}(undef, k, n)
    for t=1:k
        attended[t, :] = aux_state[t].attended_trackers
    end
    imresize(attended, ratio = (1.0 / bin, 1.0))
end

function load_attmap(trial_path::String; bin::Int64 = 4)
    chain_paths = filter(x -> occursin("jld", x), readdir(trial_path))
    chain_paths = map(x -> joinpath(trial_path, x), chain_paths)
    chains = map(extract_chain, chain_paths)
    atts = map(attention_map, chains, fill(bin, length(chains)))
    sum(atts) ./ length(chain_paths)
end

"""
    z scored attention maps from an experiment results folder
"""
function load_attmaps(exp_path::String; bin::Int64 = 4)
    trials = filter(isdir, readdir(exp_path; join = true))
    attmap = load_attmap(first(trials), bin=bin)
    attmaps = Array{Float64}(undef, length(trials), size(attmap)...)
    ts = 1:size(attmap, 1)
    results = []
    for (i, trial) in enumerate(trials)
        print("getting attmap for trial $i \r")
        attmaps[i,:,:] = load_attmap(trial)
        df = DataFrame(t = ts,
                       tracker_1 = attmaps[i, :, 1],
                       tracker_2 = attmaps[i, :, 2],
                       tracker_3 = attmaps[i, :, 3],
                       tracker_4 = attmaps[i, :, 4])
        df[!, :trial] .= i
        push!(results, df)
    end
    print("done                               \n")
    mu = mean(attmaps)
    std = Statistics.std(attmaps)
    attmaps = (attmaps .- mu)/std
    results = vcat(results...)
    CSV.write(joinpath(exp_path, "attention.csv"), results)
    return attmaps
end

"""
    returns boolean attention maps for probe placement
    n - number of difficulty levels (quantiles)
"""
function bool_attmaps(attmaps::Array{Float64}, n_quantiles::Int)
    n_trials = size(attmaps, 1)
    quantiles = nquantile(collect(Iterators.flatten(attmaps)), n_quantiles)
    b_ams = Array{Array{Bool}}(undef, n_trials, n_quantiles)
    for i=1:n_trials
        for j=1:n_quantiles
            b_ams[i,j] = (quantiles[j] .< attmaps[i,:,:]) .& (attmaps[i,:,:] .< quantiles[j+1])
            # b_ams[i,j][1:2,:] .= false
            # b_ams[i,j][end-1:end,:] .= false
        end
    end
    b_ams
end


function plot_attmap(att::Matrix{Float64}, path::String)
    p = Gadfly.spy(att', Guide.xlabel("Time"), Guide.ylabel("Tracker"))
    p |> PNG(path)
end

function compare_att(a::String, b::String, i::Int64;
                             k::Int64 = 2)
    a_i = load_attmap("$a/$i")
    b_i = load_attmap("$b/$i")
    collect((a_i .- b_i)')
end

function compare_experiments(a::String, b::String;
                             n::Int64 = 128,
                             bin::Int64 = 4)
    out = "/experiments/$(a)_vs_$(b)"
    isdir(out) || mkdir(out)
    a_path = "/experiments/$a"
    b_path = "/experiments/$b"
    results = []
    for i = 1:n
        ds = compare_att(a_path, b_path, i)
        df = DataFrame(t = 1:size(ds, 2),
                       tracker_1 = ds[1, :],
                       tracker_2 = ds[2, :],
                       tracker_3 = ds[3, :],
                       tracker_4 = ds[4, :])
        df[!, :trial] .= i
        push!(results, df)
        # plot_attmap(ds, joinpath(out, "$i_att.png"))
        # CSV.write(joinpath(out, "$i.csv"), DataFrame(ds[ps,:]),
        #           writeheader=false)
    end
    results = vcat(results...)
    CSV.write(joinpath(out, "attention.csv"), results)
end
