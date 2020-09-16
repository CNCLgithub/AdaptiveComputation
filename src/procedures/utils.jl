export extract_tracker_positions,
        extract_tracker_velocities,
        extract_assignments,
        extract_tracker_masks,
        extract_pmbrfs_params,
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

    tracker_positions = Array{Float64}(undef, length(trackers), 3)
    for i=1:length(trackers)
        tracker_positions[i,:] = trackers[i].pos
    end

    tracker_positions = reshape(tracker_positions, (1,1,size(tracker_positions)...))
    return tracker_positions
end

function extract_tracker_velocities(trace::Gen.Trace)
    (init_state, states) = Gen.get_retval(trace)

    trackers = states[end].graph.elements

    tracker_velocities = Array{Float64}(undef, length(trackers), 2)
    for i=1:length(trackers)
        tracker_velocities[i,:] = trackers[i].vel
    end

    tracker_velocities = reshape(tracker_velocities, (1,1,size(tracker_velocities)...))
    return tracker_velocities
end

function extract_assignments(trace::Gen.Trace)
    t, motion, gm = Gen.get_args(trace)
    ret = Gen.get_retval(trace)
    pmbrfs = ret[2][t].rfs
    record = AssociationRecord(5)
    xs = get_choices(trace)[:kernel => t => :masks]
    Gen.logpdf(rfs, xs, pmbrfs, record)
    (record.table, record.logscores)
end

function extract_tracker_masks(trace::Gen.Trace)
    t, motion, gm = Gen.get_args(trace)
    ret = Gen.get_retval(trace)
    pmbrfs = ret[2][t].rfs
    
    tracker_masks = Vector{Array{Float64,2}}(undef, gm.n_trackers)

    for i=1:gm.n_trackers
        tracker_masks[i] = first(GenRFS.args(pmbrfs[1+i]))
    end

    tracker_masks = reshape(tracker_masks, (1,1,size(tracker_masks)...))

    return tracker_masks
end

function extract_causal_graph(trace::Gen.Trace)
    t, motion, gm = Gen.get_args(trace)
    ret = Gen.get_retval(trace)
    causal_graph = [ret[2][t].graph]
    causal_graph = reshape(causal_graph, (1,1,size(causal_graph)...))
end
