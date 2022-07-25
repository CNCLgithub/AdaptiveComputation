export extract_tracker_positions,
    extract_tracker_velocities,
    extract_assignments,
    extract_chain

using JLD2
using DataFrames
using Gen_Compose:SeqPFChain

function digest_auxillary(c::SeqPFChain)
    deepcopy(c.auxillary)
end

function extract_parents(c::SeqPFChain)
    deepcopy(c.state.parents)
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

function digest_tracker_positions(c::SeqPFChain)
    np = length(c.state.traces)
    nt = @> (c.state.traces) begin
        first
        get_retval # (init, rest)
        last # (state1, state2,...)
        last # state_n
        get_objects
        length # nt
    end
    pos = Array{Float64, 3}(undef, np, nt, 2)
    for i = 1:np
        (_, states) = Gen.get_retval(c.state.traces[i])
        trackers = @> states last get_objects
        for j = 1:nt
            pos[i, j, :] = trackers[j].pos
        end
    end
    mean(pos, dims = 1) # avg position for each tracker
end



function td_accuracy(td::Dict{Int64, Float64}; nt::Int64 = 4)
    ws = Vector{Float64}(undef, nt)
    @inbounds for k = 1:nt
        ws[k] = exp(td[k])
    end
    ws
end
function td_accuracy(td::Dict{BitVector, Float64}; k::Int64 = 4)
    denom = log(k)
    mass = -Inf
    for (key, val) in td
        w = log(sum(key[1:k])) - denom + get(td, key, -Inf)
        mass = logsumexp(mass, w)
    end
    exp(mass - logsumexp(collect(values(td))))
end

function extract_td_accuracy(c::SeqPFChain, ntargets::Int64)
    # particles at last frame of inference
    @unpack state = c
    traces = sample_unweighted_traces(state, length(state.traces))
    @>> traces begin
        map(td_flat) # traces
        map(x -> td_accuracy(x; nt = ntargets))
        mean        # average across traces
    end
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
