using Gen_Compose:PFChain

function digest_auxillary(c::PFChain)
    deepcopy(c.auxillary)
end

function digest_tracker_positions(c::PFChain)
    # number of particles and targets
    np = length(c.state.traces)
    nt = @> (c.state.traces) begin
        first
        get_retval # (init, rest)
        first # init
        get_objects
        length # num targets
    end
    traces = sample_unweighted_traces(c.state, np)
    # traces = c.state.traces
    pos = Array{Float64, 3}(undef, np, nt, 2)
    for i = 1:np
        (_, states) = Gen.get_retval(traces[i])
        trackers = @> states last get_objects
        for j = 1:nt
            pos[i, j, :] = get_pos(trackers[j])
        end
    end
    avg_pos = mean(pos, dims = 1) # trackers x 2
    sd_pos = std(pos, mean = avg_pos, dims = 1) # trackers x 2
    (avg = avg_pos, sd = sd_pos)
end

function digest_td_accuracy(c::PFChain)
    # particles at last frame of inference
    @unpack state = c
    traces = sample_unweighted_traces(state, length(state.traces))
    @>> traces begin
        map(td_assocs) # traces
        mean        # average across traces
    end
end
