struct State
    graph::CausalGraph
    rfs::RFSElements{Array}
    flow_masks::Union{Nothing, FlowMasks}
end

@gen function sample_init_tracker(init_pos_spread::Real)::Dot
    
    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)

    vx = 0.0
    vy = 0.0

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    return Dot([x,y,z], [vx, vy])
end

init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(gm::GMParams)
    trackers_gm = fill(gm.init_pos_spread, gm.n_trackers)
    trackers = @trace(init_trackers_map(trackers_gm), :trackers)
    trackers = collect(Object, trackers)
    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)
    pmbrfs = RFSElements{Array}(undef, 0)

    if gm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, gm.n_trackers, gm.fmasks_n)
        for i=1:gm.n_trackers
            for j=1:gm.fmasks_n
                fmasks[i,j] = zeros(gm.img_height, gm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               gm.fmasks_decay_function)
    else
        flow_masks = nothing
    end
    
    State(graph, pmbrfs, flow_masks)
end

