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

    traces = chain.state.traces
    # TODO: generalize across kernel states
    pf_st = Matrix{CausalGraph}(undef, np, k)
    for i = 1:np
        # pf_st[i, :] = map(world, last(get_retval((traces[i]))))
        pf_st[i, :] = @>> get_retval(traces[i]) last map(world)
    end

    df = DataFrame(
                   frame = Int64[],
                   tracker = Int64[],
                   cycles = Int64[],
                   pred_x = Float64[],
                   pred_y = Float64[])
    cgs = map(world, chain.
    for frame = 1:steps
        cycles_per_part = collect(values(aux_state[frame].allocated))
        # assuming all particles are aligned wrt tracker ids for now
        cpt = isempty(cycles_per_part) ? zeros(n_targets) : sum(cycles_per_part)
        np = size(pf_st, 1)
        positions = Matrix{Float64}(undef, 3, n_targets, np)
        for p = 1:np
            trackers = get_objects(pf_st[p, frame], Dot)
            for i = 1:n_targets 
                positions[:, i, p] = trackers[i].pos
            end
        end
        for i = 1:n_targets
            px, py, _ = mean(positions[:, i, :], dims = 1)
            push!(df, (frame, i, cpt[i], px, py))
        end
    end
    return df
end
