using Gen_Compose:SeqPFChain

function digest_auxillary(c::SeqPFChain)
    deepcopy(c.auxillary)
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

function extract_td_accuracy(c::SeqPFChain)
    # particles at last frame of inference
    @unpack state = c
    traces = sample_unweighted_traces(state, length(state.traces))
    @>> traces begin
        map(td_assocs) # traces
        mean        # average across traces
    end
end
