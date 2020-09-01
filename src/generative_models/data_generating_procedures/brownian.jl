export dgp

using Setfield

_dgp(k::Int, gm::GMMaskParams, motion::AbstractDynamicsModel) = error("not implemented")
_dgp(k::Int, gm::GMMaskParams, motion::BrownianDynamicsModel) = gm_brownian_pos(k, motion, gm)
# TODO : implement gm_cbm_pos
# _dgp(k::Int, gm::GMMaskParams, motion::ConstrainedBDM) = ...
_dgp(k::Int, gm::GMMaskParams, motion::ISRDynamics) = gm_isr_pos(k, motion, gm)

function dgp(k::Int, params::GMMaskParams,
             motion::AbstractDynamicsModel;
             generate_masks=true)

    # new params with all dots having state for data generation
    gm = deepcopy(params)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    
    # running generative model on just positions (no need to go to masks)
    init_state, states = _dgp(k, gm, motion)

    num_dots = gm.n_trackers

    # initial positions and positions over time will be returned
    # from this generative process
    init_positions = Array{Float64}(undef, num_dots, 3)
    init_vels = Array{Float64}(undef, num_dots, 2)
    positions = Vector{Array{Float64}}(undef, k)

    for i=1:num_dots
        init_positions[i,:] = init_state.graph.elements[i].pos
        init_vels[i,:] = init_state.graph.elements[i].vel
    end
    
    for t=1:k
        dots = states[t].graph.elements
        pos_t = Array{Float64}(undef, length(dots), 3)
        for i=1:num_dots
            pos_t[i,:] = dots[i].pos
        end
        positions[t] = pos_t
    end
    
    if generate_masks
        masks = get_masks(positions, params.dot_radius, params.img_height,
                          params.img_width, params.area_height, params.area_width)
    else
        masks = nothing
    end

    return init_positions, init_vels, masks, positions
end
