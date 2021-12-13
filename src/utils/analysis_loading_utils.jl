export read_json, merge_trial, merge_experiment

using CSV
using CSV: write
using JSON

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


function merge_trial(trial_dir::String)::DataFrame
    @>> trial_dir begin
        readdir(; join = true)
        filter(x -> occursin("csv", x))
        map(DataFrame ∘ CSV.File)
        x -> vcat(x...)
    end
end

function merge_experiment(exp_path::String)
    @>> exp_path begin
        readdir(;join = true)
        filter(isdir)
        map(merge_trial)
        x -> vcat(x...)
        write("$(exp_path).csv")
    end
    return nothing
    # trials = filter(isdir, readdir(exp_path, join = true))
    # df = vcat(map(merge_trial, trials)...)
    # CSV.write("$(exp_path).csv", df)
    # return nothing
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

function chain_performance(chain, path;
                           n_targets = 4)
    dg = extract_digest(path)
    aux_state = dg[:, :auxillary]
    # causal graphs at the end of inference
    td_acc = extract_td_accuracy(chain, n_targets)
    df = DataFrame(
                   tracker = 1:n_targets,
                   td_acc = td_acc)
    return df
end

function chain_attention(chain, path;
                         n_targets = 4,
                         n_objects = 8)
    dg = extract_digest(path)
    aux_state = dg[:, :auxillary]

    steps = length(aux_state)
    # cycles = 0
    df = DataFrame(
                   frame = Int64[],
                   tracker = Int64[],
                   cycles = Int64[])
    for frame = 1:steps
        cycles_per_part = collect(values(aux_state[frame].allocated))
        # assuming all particles are aligned wrt tracker ids for now
        cpt = isempty(cycles_per_part) ? zeros(n_targets) : sum(cycles_per_part)
        for i = 1:n_targets
            push!(df, (frame, i, cpt[i]))
        end
    end
    return df
end
