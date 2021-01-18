struct ReceptiveFieldsState
    graph::CausalGraph{Object, SimpleGraph}
    rfs_vec::Vector{RFSElements{Array}}
    flow_masks::Union{Nothing, FlowMasks}
end

@gen function sample_init_receptive_fields_state(gm::GMMaskParams)
    trackers_gm = fill(gm.init_pos_spread, gm.n_trackers)
    exp0 = fill(gm.exp0, gm.n_trackers)
    trackers = @trace(init_trackers_map(trackers_gm, exp0), :trackers)
    trackers = collect(Object, trackers)

    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)

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
    
    ReceptiveFieldsState(graph, rfs_vec, flow_masks)
end

@gen function sample_masks(pmbrfs)
    @trace(rfs(pmbrfs), :masks) 
end

receptive_fields_map = Gen.Map(sample_masks)

@gen static function receptive_fields_kernel(t::Int,
                                      prev_state::ReceptiveFieldsState,
                                      dynamics_model::AbstractDynamicsModel,
                                      receptive_fields::Vector{AbstractReceptiveField},
                                      prob_threshold::Float64,
                                      gm::GMMaskParams)
    # using ISR Dynamics
    new_graph = @trace(inertial_update(dynamics_model, prev_state.graph), :dynamics)
    objects = new_graph.elements
    rfs_vec, flow_masks = get_rfs_vec(receptive_fields, objects, prob_threshold, gm, flow_masks=prev_state.flow_masks)
    @trace(receptive_fields_map(rfs_vec), :receptive_fields)

    # returning this to get target designation and assignment
    new_state = ReceptiveFieldsState(new_graph, rfs_vec, prev_state.flow_masks)
    return new_state
end

receptive_fields_chain = Gen.Unfold(receptive_fields_kernel)

@gen static function gm_receptive_fields(k::Int,
                                         dynamics_model::AbstractDynamicsModel,
                                         gm::GMMaskParams,
                                         receptive_fields::Vector{AbstractReceptiveField},
                                         prob_threshold::Float64)
    init_state = @trace(sample_init_receptive_fields_state(gm), :init_state)
    states = @trace(receptive_fields_chain(k, init_state, dynamics_model, receptive_fields, prob_threshold, gm), :kernel)

    result = (init_state, states)
    return result
end



export gm_receptive_fields
