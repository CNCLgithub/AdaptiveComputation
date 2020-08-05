using CSV
using DataFrames
using ArgParse

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

function attention_maps(experiment::String)
    for i = 1:120
        trial_path = "$(experiment)/($i)"
        chain_paths = filter(x -> occursin("jld", x), readdir(trial_path))
        chains = map(extract_chain, chain_paths)
        atts = mean(map(attention_map, chains), dims = 1)
        out_path = joinpath(trial_path, "$(i)_att.csv")
        CSV.write(out_path, DataFrame(atts), writeheader=false)
    end
end
