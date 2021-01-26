
# TODO maybe make faster
function get_rendered_dots(cg, targets)
    dots = []
    for e in cg.elements
        if isa(e, Polygon)
            dots = [dots; e.dots]
        else
            push!(dots, e)
        end
    end
    
    dots[targets]
end

@gen function sample_init_hgm_receptive_fields_state(hgm::HGMParams)
    hgm_trackers = fill(hgm, hgm.n_trackers)
    trackers = @trace(init_dot_or_polygon_map(hgm_trackers), :trackers)
    trackers = collect(Object, trackers)

    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)

    rfs_vec = RFSElements{Array}(undef, 0)
    
    #n_rendered_dots = length(get_rendered_dots(graph, hgm.targets))
    n_rendered_dots = sum(hgm.targets)

    if hgm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, n_rendered_dots, hgm.fmasks_n)
        for i=1:n_rendered_dots
            for j=1:hgm.fmasks_n
                fmasks[i,j] = zeros(hgm.img_height, hgm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               hgm.fmasks_decay_function)
    else
        flow_masks = nothing
    end

    ReceptiveFieldsState(graph, rfs_vec, flow_masks)
end



@gen static function hgm_receptive_fields_kernel(t::Int,
#@gen function hgm_receptive_fields_kernel(t::Int,
                                      prev_state::ReceptiveFieldsState,
                                      dynamics_model::AbstractDynamicsModel,
                                      receptive_fields::Vector{RectangleReceptiveField},
                                      prob_threshold::Float64,
                                      hgm::HGMParams)
    # using ISR Dynamics
    new_graph = @trace(hgm_inertia_update(dynamics_model, prev_state.graph), :dynamics)
    dots = collect(Object, get_rendered_dots(new_graph, hgm.targets))
     
    rfs_vec, flow_masks = get_rfs_vec(receptive_fields, dots, prob_threshold, hgm, flow_masks=prev_state.flow_masks)
    @trace(receptive_fields_map(rfs_vec), :receptive_fields)

    # returning this to get target designation and assignment
    new_state = ReceptiveFieldsState(new_graph, rfs_vec, flow_masks)
    return new_state
end

hgm_receptive_fields_chain = Gen.Unfold(hgm_receptive_fields_kernel)

@gen static function hgm_receptive_fields(k::Int,
#@gen function hgm_receptive_fields(k::Int,
                                   dynamics_model::AbstractDynamicsModel,
                                   hgm::HGMParams,
                                   receptive_fields::Vector{RectangleReceptiveField},
                                   prob_threshold::Float64)

    init_state = @trace(sample_init_hgm_receptive_fields_state(hgm), :init_state)
    states = @trace(hgm_receptive_fields_chain(k, init_state, dynamics_model, receptive_fields, prob_threshold, hgm), :kernel)

    result = (init_state, states)
    return result
end


export hgm_receptive_fields
