using CSV
using Statistics
using DataFrames

function attention_map(c::Dict)
    aux_state = c["aux_state"]
    k = length(aux_state)
    n = length(first(aux_state).attended_trackers)
    attended = Matrix{Float64}(undef, k, n)
    for t=1:k
        attended[t, :] = aux_state[t].attended_trackers
    end
    attended
end

function attention_maps(experiment::String; n::Int = 128)
    for i = 1:n
        trial_path = "$(experiment)/$i"
        chain_paths = filter(x -> occursin("jld", x), readdir(trial_path))
        chain_paths = map(x -> joinpath(trial_path, x), chain_paths)
        chains = map(extract_chain, chain_paths)
        atts = map(attention_map, chains)
        atts = sum(atts) ./ length(chain_paths) 
        display(atts)
        out_path = joinpath(trial_path, "$(i)_att.csv")
        CSV.write(out_path, DataFrame(atts), writeheader=false)
    end
end
