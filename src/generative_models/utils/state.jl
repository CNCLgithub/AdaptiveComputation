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

    return Dot([x,y,z], [vx, vy])
end

init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(gm::GMParams, dm)
    trackers_gm = fill(gm, gm.n_trackers)
    current_state = @trace(Gen.Map(sample_init_tracker)(trackers_gm), :trackers)
    cg = process_temp_state(current_state, gm, dm)

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
    
    State(cg, pmbrfs, flow_masks)
end

@gen function sample_init_receptive_fields_state(gm::GMParams, dm)
    trackers_gm = fill(gm, gm.n_trackers)
    current_state = @trace(Gen.Map(sample_init_tracker)(trackers_gm), :trackers)
    cg = process_temp_state(current_state, gm, dm)

    #pmbrfs = RFSElements{Array}(undef, 0)
    rfs_vec = Vector{RFSElements{Array}}(undef, 0)

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
    
    RFState(cg, rfs_vec, flow_masks)
end

