export read_json,
        analysis_load_trial,
        analysis_load_results

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

function analysis_load_trial(trial_dir::String)
    runs = filter(x -> occursin("jld", x),
                  readdir(trial_dir, join = true))
    n_runs = length(runs)

    performance = Array{Float64}(undef, n_runs)
    compute = zeros(n_runs)
    pred_target = zeros(n_runs, 8)

    for (run, chain) in enumerate(runs)

        # reading the file
        extracted = extract_chain(chain)
        
        # getting performance for this run
        final_log_scores = extracted["log_scores"][end,:]
        final_assignments = extracted["unweighted"][:assignments][end,:]
        performance[run] = td_accuracy(final_assignments)

        # getting pred_target
        weighted_assignments = extracted["weighted"][:assignments][end,:]
        map_idx = argmax(final_log_scores)
        assocs, ls = weighted_assignments[map_idx]
        assocs = map(x -> vcat(x[2:end]...), assocs)
        pred_target[run, first(assocs)] .= 1.0

        # getting the compute for this run
        aux_state = extracted["aux_state"]
        for i=1:length(aux_state)
            attempts = aux_state[i].attempts
            compute[run] += sum(attempts)
        end
    end

    return Dict("performance" => performance,
                "compute" => compute,
                "pred_target" => pred_target)
end


function analysis_load_results(dir::String)
    println("reading the trial files from $dir...")
	trials = readdir(dir)

    n_trials = length(trials)
    n_runs = length(readdir(joinpath(dir, "1")))

    performance = Array{Float64}(undef, n_trials, n_runs)
    compute = zeros(n_trials, n_runs)
    # for each dot whether it's predicted to be a target
    pred_target = zeros(n_trials, n_runs, 8)

    for trial=1:n_trials
        print("trial $trial / $(length(trials))\r")

        trial_dir=joinpath(dir, "$trial")
        trial_results = analysis_load_trial(trial_dir)
        performance[trial,:] = trial_results["performance"]
        compute[trial,:] = trial_results["compute"]
        pred_target[trial,:,:] = trial_results["pred_target"]

    end
    
    return Dict("performance" => performance,
                "compute" => compute,
                "pred_target" => pred_target)
end
