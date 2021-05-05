# classic state
struct State
    cg::CausalGraph
    rfs::RFSElements{Array}
    flow_masks::Union{Nothing, FlowMasks}
end

# receptive fields state
struct RFState
    cg::CausalGraph
    rfs_vec::Vector{RFSElements{Array}}
    flow_masks::Union{Nothing, FlowMasks}
end

@gen function sample_init_tracker(gm::AbstractGMParams)::Dot
    @unpack area_width, area_height, dot_radius = gm
    
    x = @trace(uniform(-area_width/2 + dot_radius, area_width/2 - dot_radius), :x)
    y = @trace(uniform(-area_height/2 + dot_radius, area_height/2 - dot_radius), :y)

    vx = 0.0
    vy = 0.0

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    return Dot([x,y,z], [vx, vy], dot_radius)
end

@gen function sample_init_state(gm::GMParams, dm)
    trackers_gm = fill(gm, gm.n_trackers)
    current_state = @trace(Gen.Map(sample_init_tracker)(trackers_gm), :trackers)

    cg = process_temp_state(current_state, gm, dm)
    pmbrfs = RFSElements{Array}(undef, 0)
    flow_masks = gm.fmasks ? FlowMasks(gm.n_trackers, gm) : nothing
    
    State(cg, pmbrfs, flow_masks)
end

@gen function sample_init_receptive_fields_state(gm::GMParams, dm)
    trackers_gm = fill(gm, gm.n_trackers)
    current_state = @trace(Gen.Map(sample_init_tracker)(trackers_gm), :trackers)

    cg = process_temp_state(current_state, gm, dm)
    rfs_vec = Vector{RFSElements{Array}}(undef, 0)
    flow_masks = gm.fmasks ? FlowMasks(gm.n_trackers, gm) : nothing
    
    RFState(cg, rfs_vec, flow_masks)
end

