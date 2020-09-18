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
    sum(map(x -> length(intersect(x, 1:4)/4), assocs) .* weights)
end

function td_accuracy(particles)
    accs = map(_td_accuracy, zip(particles...)...)
    mean(accs)
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
