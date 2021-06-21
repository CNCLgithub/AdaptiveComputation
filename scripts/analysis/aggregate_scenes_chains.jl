using CSV
using DataFrames


function aggregate_scenes_chains(path, out)
    dfs = []

    scenes = readdir(path)
    for scene in scenes
        chains = readdir(joinpath(path, scene))
        for chain in chains
            fn = joinpath(path, scene, chain)
            df = CSV.File(fn) |> DataFrame
            push!(dfs, df)
        end
    end
    
    @show vcat(dfs...) 

    CSV.write(out, vcat(dfs...))
end


path = "/experiments/exp1_difficulty_target_designation/"
out = "/experiments/exp1_difficulty_target_designation.csv"

aggregate_scenes_chains(path, out)
