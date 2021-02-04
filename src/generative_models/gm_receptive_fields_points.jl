struct ReceptiveFieldsPointsState
    graph::CausalGraph{Object, SimpleGraph}
    rfs_vec::Vector{RFSElements{Array}}
end

@with_kw struct GMPointParams <: AbstractGMParams
    n_trackers::Int = 4
    distractor_rate::Real = 4.0
    init_pos_spread::Real = 300.0
    
    area_height::Int = 800
    area_width::Int = 800
    
    obs_noise::Float64 = 1.0

    # rfs parameters
    record_size::Int = 100 # number of associations

    # some extra stuff for rendering
    img_width = 600
    img_height = 600
    dot_radius = 20
end

function load(::Type{GMPointParams}, path; kwargs...)
    GMPointParams(;read_json(path)..., kwargs...)
end
function load(::Type{GMPointParams}, path::String)
    GMPointParams(;read_json(path)...)
end

@gen function sample_init_receptive_fields_points_state(gm::GMPointParams)
    trackers_gm = fill(gm.init_pos_spread, gm.n_trackers)
    exp0 = fill(false, gm.n_trackers)
    trackers = @trace(init_trackers_map(trackers_gm, exp0), :trackers)
    trackers = collect(Object, trackers)

    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)
    rfs_vec = Vector{RFSElements{Array}}(undef, 0)
    
    ReceptiveFieldsPointsState(graph, rfs_vec)
end

@gen function sample_points(pmbrfs)
    @trace(rfs(pmbrfs), :points) 
end

receptive_fields_points_map = Gen.Map(sample_points)

@gen static function receptive_fields_points_kernel(t::Int,
                                      prev_state::ReceptiveFieldsPointsState,
                                      dynamics_model::InertiaModel,
                                      receptive_fields::Vector{RectangleReceptiveField},
                                      gm::GMPointParams)

    # using ISR Dynamics
    new_graph = @trace(inertial_update(dynamics_model, prev_state.graph), :dynamics)
    objects = new_graph.elements

    rfs_vec = get_rfs_vec_points(receptive_fields, objects, gm)
    # rfs_vec, flow_masks = get_rfs_vec(receptive_fields, objects, prob_threshold, gm, flow_masks=prev_state.flow_masks)
    @trace(receptive_fields_points_map(rfs_vec), :receptive_fields)

    # returning this to get target designation and assignment
    new_state = ReceptiveFieldsPointsState(new_graph, rfs_vec)
    return new_state
end

receptive_fields_points_chain = Gen.Unfold(receptive_fields_points_kernel)

@gen static function gm_receptive_fields_points(k::Int,
                                         dynamics_model::InertiaModel,
                                         gm::GMPointParams,
                                         receptive_fields::Vector{RectangleReceptiveField})
    init_state = @trace(sample_init_receptive_fields_points_state(gm), :init_state)
    states = @trace(receptive_fields_points_chain(k, init_state, dynamics_model, receptive_fields, gm), :kernel)

    result = (init_state, states)
    return result
end


export gm_receptive_fields_points, GMPointParams
