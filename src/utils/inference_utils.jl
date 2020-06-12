export extract_tracker_positions,
        extract_assignments,
        extract_chain

function extract_chain(r::Gen_Compose.SequentialChain)
    weighted = []
    unweighted = []
    log_scores = []
    ml_est = []
    states = []
    for t = 1:length(r.buffer)
        state = r.buffer[t]
        push!(weighted, state["weighted"])
        push!(unweighted, state["unweighted"])
        push!(log_scores, state["log_scores"])
        push!(ml_est, state["ml_est"])
    end
    weighted = merge(vcat, weighted...)
    unweighted = merge(vcat, unweighted...)
    log_scores = vcat(log_scores...)
    extracts = Dict("weighted" => weighted,
                    "unweighted" => unweighted,
                    "log_scores" => log_scores,
                    "ml_est" => ml_est)
    return extracts
end

function extract_tracker_positions(trace::Gen.Trace)
    (init_state, states) = Gen.get_retval(trace)

    trackers = states[end].trackers

    tracker_positions = Array{Float64}(undef, length(trackers), 2)
    for i=1:length(trackers)
        tracker_positions[i,1] = trackers[i].pos[1]
        tracker_positions[i,2] = trackers[i].pos[2]
    end

    tracker_positions = reshape(tracker_positions, (1,1,size(tracker_positions)...))
    return tracker_positions
end

function extract_assignments(trace::Gen.Trace)
    t, params = Gen.get_args(trace)
    ret = Gen.get_retval(trace)

    #optics = trace[:states => t => :optics]
    #tds, As, td_weights = get_td_A(pmbrfs, optics, ret[2][end].ppp_params, ret[2][end].mbrfs_params)
     
    saved_td = ret[2][t].pmbrfs_params.saved_td
    tds, As, td_weights = saved_td.td, saved_td.A, saved_td.ll

    A = tds[1][invperm(As[1])] # this is to get the old sense of assignment, i.e. mapping from trackers to observations

    A = reshape(A, (1,1,size(A)...))

    return A
end
