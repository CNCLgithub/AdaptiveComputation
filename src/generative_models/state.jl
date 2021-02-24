struct State
    graph::CausalGraph
    rfs::RFSElements{Array}
    flow_masks::Union{Nothing, FlowMasks}
end
