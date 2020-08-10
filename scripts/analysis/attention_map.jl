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

function attention_map(c::Dict; bin::Int64 = 4)
    aux_state = c["aux_state"]
    k = length(aux_state)
    n = length(first(aux_state).attended_trackers)
    attended = Matrix{Float64}(undef, k, n)
    for t=1:k
        attended[t, :] = aux_state[t].attended_trackers
    end
    imresize(attended, ratio = (1.0 / bin, 1.0))
end

function load_attmap(trial_path::String)
    chain_paths = filter(x -> occursin("jld", x), readdir(trial_path))
    chain_paths = map(x -> joinpath(trial_path, x), chain_paths)
    chains = map(extract_chain, chain_paths)
    atts = map(attention_map, chains)
    sum(atts) ./ length(chain_paths)
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
        ds = compare_att(a, b, i)
        df = DataFrame(t = 1:size(ds, 2),
                       tracker_1 = ds[1, :],
                       tracker_2 = ds[2, :],
                       tracker_3 = ds[3, :],
                       tracker_4 = ds[4, :])
        df.trial = i
        push!(results, df)
        # plot_attmap(ds, joinpath(out, "$i_att.png"))
        # CSV.write(joinpath(out, "$i.csv"), DataFrame(ds[ps,:]),
        #           writeheader=false)
    end
    results = vcat(results...)
    CSV.write(joinpath(out, "attention.csv"), results)
end
