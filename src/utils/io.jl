export read_json, extract_digest, merge_trial, merge_experiment

using CSV
using CSV: write
using JSON
using JLD2
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

"""
    simplified target designation using the Hungarian algorithm
    (or a simple version of the Hungarian algorithm)
"""
function get_simplified_target_designation(cg, gt_cg)
    pos = @>> get_objects(cg, Dot) map(x->x.pos)
    gt_pos = @>> get_objects(gt_cg, Dot) map(x->x.pos)
    
    n_trackers = @>> get_objects(cg, Dot) length
    n_dots = @>> get_objects(gt_cg, Dot) length

    inds = Iterators.product(1:n_trackers, 1:n_dots)
    distances = @>> inds map(i -> norm(pos[i[1]] - gt_pos[i[2]]))
    td = []
    for i=1:n_trackers
        perm = sortperm(distances[i,:])
        for j in perm
            if !(j in td)
                push!(td, j)
                break
            end
        end
    end

    return td
end

function chain_performance(chain, dg;
                           n_targets = 4)
    aux_state = dg[:, :auxillary]
    # causal graphs at the end of inference
    td_acc = extract_td_accuracy(chain, n_targets)
    df = DataFrame(
                   tracker = 1:n_targets,
                   td_acc = td_acc)
    return df
end

function chain_attention(chain, dg;
                         n_targets = 4,
                         n_objects = 8)
    aux_state = dg[:, :auxillary]

    steps = length(aux_state)
    # cycles = 0

    traces = chain.state.traces
    np = length(traces)
    df = DataFrame(
        frame = Int64[],
        tracker = Int64[],
        importance = Float64[],
        cycles = Float64[],
        pred_x = Float64[],
        pred_y = Float64[])
    for frame = 1:steps
        arrousal = aux_state[frame].arrousal
        importance = aux_state[frame].importance
        cycles_per_latent =  arrousal .* importance
        positions = dg[frame, :positions]
        for i = 1:n_targets
            px, py = positions[1, i, :]
            push!(df, (frame, i,
                       importance[i],
                       cycles_per_latent[i],
                       px, py))
        end
    end
    return df
end
