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
    map(x -> x.pos[1:2], cg.elements)
end

function analyze_chain_receptive_fields(chain,
                                        n_trackers = 4,
                                        n_dots = 8,
                                        receptive_fields = nothing,
                                        masks_end = nothing)

    # reading the file of memory buffer
    extracted = extract_chain(chain)
    
    df = DataFrame(frame = Int64[],
                   tracker = Int64[],
                   attention = Float64[],
                   td_acc = Float64[])

    aux_state = extracted["aux_state"]
    correspondence = extracted["unweighted"][:assignments] # t x particle
    causal_graphs = extracted["unweighted"][:causal_graph] # t x particle

    display(correspondence[1,1,:])
    
    # just average td_acc across particles for the last step
    n_particles = size(correspondence)[2]
    td = @>> 1:n_particles begin
        map(i -> MOT.get_target_designation(n_trackers, correspondence[1,i,:], masks_end, receptive_fields))
    end
    # taking the top designation for each particle (or sampling random if top logscore is Inf)
    display(td[1][1:5])
    top_td = @>> td map(x-> isinf(x[1][2]) ? collect(combinations(n_dots, n_trackers)) : x[1][1])
    display(top_td)
    td_acc = @>> top_td map(x -> length(intersect(x, collect(1:n_trackers)))/n_trackers) mean
    
    println(td_acc)

    for frame = 1:size(causal_graphs)[1], tracker = 1:n_trackers
        att = aux_state[frame].attended_trackers[tracker]
        #probs = td[frame, particle][tracker]
        #pos = positions[frame, particle][tracker]
        push!(df, (frame, tracker, att, td_acc))
    end
    return df
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
    display(df)
    CSV.write(joinpath(exp_path, "merged_results.csv"), df)
    return nothing
end
