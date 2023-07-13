using CSV
using JSON
using DataFrames

function main()
    src_path = "/spaths/datasets/exp3.json"
    out_path = "/spaths/datasets/exp3_design.csv"
    dataset = JSON.parsefile(src_path)

    df = DataFrame(scene = Int64[],
                   ntargets = Int64[],
                   vel = Float64[])

    nscenes = length(dataset)
    for i = 1:nscenes
        auxdata = dataset[i]["aux_data"]
        push!(df, (scene = i,
                   ntargets = sum(auxdata["targets"]),
                   vel = auxdata["vel"]))
    end

    CSV.write(out_path, df)
end


main();
