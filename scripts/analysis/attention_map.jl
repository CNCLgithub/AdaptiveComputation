using MOT
using CSV
using Statistics
using DataFrames
using Gadfly
Gadfly.push_theme(Theme(background_color = colorant"white"))
using Compose
import Cairo
using ImageTransformations:imresize

function attention_map(c::Dict; bin::Int64 = 3)
    aux_state = c["aux_state"]
    k = length(aux_state)
    n = length(first(aux_state).attended_trackers)
    attended = Matrix{Float64}(undef, k, n)
    for t=1:k
        attended[t, :] = aux_state[t].attended_trackers
    end
    imresize(attended, ratio = (1.0 / bin, 1.0))
end

function load_attmap(path::String)
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
    ds = a_i .- b_i
    d = map(norm, eachrow(ds))
    ps = take(sortperm(d, rev = true), 2)
    ds, ps
end

function compare_experiments(a::String, b::String; n::Int64 = 128)
    out = "/experiments/$(a)_vs_$(b)"
    isdir(out) || mkdir(out)
    a_path = "/experiments/$a"
    b_path = "/experiments/$b"
    for i = 1:n
        ds, ps = compare_att(a, b, i)
        plot_attmap(ds, joinpath(out, "$i_att.png"))
        CSV.write(joinpath(out, "$i.csv"), DataFrame(ds[ps,:]),
                  writeheader=false)
    end
end
