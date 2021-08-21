export read_json, merge_trial, merge_experiment

using CSV
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


function _td_accuracy(assocs, ls)
    assocs = map(x -> vcat(x[2:end]...), assocs)
    weights = exp.(ls .- logsumexp(ls))
    sum(map(x -> length(intersect(x, 1:4))/4, assocs) .* weights)
end

function td_accuracy(particles)
    accs = map(_td_accuracy, zip(particles...)...)
    mean(accs)
end

"""
Computes a triple consisting of:
    1) The MAP target assignment
    2) The probability of that assigment
    3) The probability of the tracker being assigned to any target
"""
function tracker_probs(tr::Int64, assocs, ls::Vector{Float64})
    weights = exp.(ls .- logsumexp(ls))
    tracker_assocs = map(x -> first(x[tr+1]), assocs)
    # MAP (dc = i)
    target = first(tracker_assocs)
    target_p = first(weights)
    # p( tracker is target )
    tracker_is_trg = sum((tracker_assocs .<= 4) .* weights)
    (target, target_p, tracker_is_trg)
end

function pos_from_cgs(cg)
    @>> get_objects(cg, Dot) map(x -> x.pos[1:2])
end

function analyze_chain(chain, n_trackers::Int64 = 4)

    # reading the file of memory buffer
    extracted = extract_chain(chain)

    # setting up dataframe
    df = DataFrame(frame = Int64[],
                   particle = Int64[],
                   tracker = Int64[],
                   attention = Float64[],
                   pred_target = Int64[], # MAP target
                   prob_target = Float64[], # prob of MAP target
                   p_is_target = Float64[], # prob is any target
                   pred_x = Float64[],
                   pred_y = Float64[]
                   )

    aux_state = extracted["aux_state"]
    correspondence = extracted["unweighted"][:assignments] # t x particle
    causal_graphs = extracted["unweighted"][:causal_graph] # t x particle

    # time x particle x tracker
    positions = causal_graphs .|> pos_from_cgs
    td = correspondence .|> x -> map(y -> tracker_probs(y, x...), 1:n_trackers)
    inds = CartesianIndices(td)
    for idx in inds, tracker = 1:n_trackers
        (frame, particle) = Tuple(inds[idx])
        att = aux_state[frame].attended_trackers[tracker]
        probs = td[frame, particle][tracker]
        pos = positions[frame, particle][tracker]
        push!(df, (frame, particle, tracker, att, probs..., pos...))
    end
    return df
end

function merge_trial(trial_dir::String)
    runs = filter(x -> occursin("csv", x),
                  readdir(trial_dir, join = true))
    vcat(map(DataFrame ∘ CSV.File , runs)...)
end

function merge_experiment(exp_path::String)
    trials = filter(isdir, readdir(exp_path, join = true))
    df = vcat(map(merge_trial, trials)...)
    CSV.write("$(exp_path).csv", df)
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

function analyze_chain_receptive_fields(chain, path;
                                        n_trackers = 4,
                                        n_dots = 8,
                                        gt_cg_end = nothing)
    dg = extract_digest(path)

    df = DataFrame(frame = Int64[],
                   tracker = Int64[],
                   cycles = Float64[],
                   sensitivity = Float64[],
                   td_acc = Float64[])

    aux_state = dg[:, :auxillary]
    # causal graphs at the end of inference
    causal_graphs = map(extract_causal_graph, chain.state.traces)
    td = map(cg -> get_simplified_target_designation(cg, gt_cg_end),
             causal_graphs)

    #display(correspondence[1,1,:])
    
    # just average td_acc across particles for the last step
    target_xs = collect(1:n_trackers)
    td_acc = @>> td begin
        map(x -> length(intersect(x, target_xs))/n_trackers)
        mean
    end

    for frame = 1:length(aux_state), tracker = 1:n_trackers
        cycles = aux_state[frame].allocated[tracker]
        sens = aux_state[frame].sensitivities[tracker]
        push!(df, (frame, tracker, cycles, sens, td_acc))
    end
    return df
end
