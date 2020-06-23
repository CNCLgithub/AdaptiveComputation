export extract_tracker_positions,
        extract_assignments,
        extract_chain

using JLD2, FileIO

function extract_chain(r::String)
    v = []
    jldopen(r, "r") do data
        for i = 1:length(keys(data))
            push!(v, data["$i"])
        end
    end
    v = collect(Dict, v)
    extract_chain(v)
end

function extract_chain(r::Gen_Compose.SequentialChain)
    extract_chain(r.buffer)
end

function extract_chain(buffer::Array{T}) where T<:Dict
    extracts = T()
    k = length(buffer)
    fields = collect(keys(first(buffer)))
    for field in fields
        results = [buffer[t][field] for t = 1:k]
        if typeof(first(results)) <: Dict
            results = merge(vcat, results...)
        else
            results = vcat(results...)
        end
        extracts[field] = results
    end
    return extracts
end

function extract_tracker_positions(trace::Gen.Trace)
    (init_state, states) = Gen.get_retval(trace)

    trackers = states[end].graph.elements

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
     
    saved_td = ret[2][t].pmbrfs_params.saved_td
    tds, As, td_weights = saved_td.td, saved_td.A, saved_td.ll

    A = tds[1][invperm(As[1])] # this is to get the old sense of assignment, i.e. mapping from trackers to observations
    A = reshape(A, (1,1,size(A)...))

    return A
end
