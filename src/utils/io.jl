export read_json, extract_digest, merge_trial, merge_experiment

using CSV
using CSV: write
using JSON
using DataFrames

"""
    read_json(path)

    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    open(path, "r") do f
        global data
        data = JSON.parse(f)
    end
    
    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end

function load(::Type{T}, path::String; kwargs...) where {T}
    T(;read_json(path)..., kwargs...)
end

function extract_digest(f::String)
    df = DataFrame()
    jldopen(f, "r") do data
        steps = data["current_idx"]
        steps === 0 && return df
        @inbounds for i = 1:steps
            push!(df, data["$i"]; cols = :union)
        end
    end
    return df
end


function extract_digest(l::MemLogger)
    df = DataFrame()
    b = buffer(l)
    for i = 1:length(b)
        push!(df, b[i]; cols = :union)
    end
    return df
end


function merge_trial(trial_dir::String, report::String)::DataFrame
    @>> trial_dir begin
        readdir(; join = true)
        filter(x -> occursin(report, x))
        map(DataFrame ∘ CSV.File)
        x -> vcat(x...)
    end
end

function merge_experiment(exp_path::String;
                          report::String = "")
    @>> exp_path begin
        readdir(;join = true)
        filter(isdir)
        map(x -> merge_trial(x, report))
        x -> vcat(x...)
        write("$(exp_path)_$(report).csv")
    end
    return nothing
end
