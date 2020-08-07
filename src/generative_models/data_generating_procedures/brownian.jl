export dgp

using Setfield

function dgp(k::Int, params::GMMaskParams,
             motion::AbstractDynamicsModel)

    # new params with all dots
    # having state for data generation
    gm = deepcopy(params)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    
    # running generative model on just positions (no need to go to masks)
    if motion isa BrownianDynamicsModel
        init_state, states = gm_positions_static(k, motion, gm)
    elseif motion isa ConstrainedBDM
        init_state, states = gm_positions_cbm_static(k, motion, gm)
    elseif motion isa ISRDynamics
        init_state, states = gm_positions_isr_static(k, motion, gm)
    else
        error("unrecognized motion model")
    end


    num_dots = gm.n_trackers

    # initial positions and positions over time will be returned
    # from this generative process
    init_positions = Array{Float64}(undef, num_dots, 3)
    #init_vels = Array{Float64}(undef, num_dots, 2)
    positions = Vector{Array{Float64}}(undef, k)

    for i=1:num_dots
        init_positions[i,:] = init_state.graph.elements[i].pos
        #init_vels[i,:] = init_state.graph.elements[i].vel
    end
    
    for t=1:k
        dots = states[t].graph.elements
        pos_t = Array{Float64}(undef, length(dots), 3)
        for i=1:num_dots
            pos_t[i,:] = dots[i].pos
        end
        positions[t] = pos_t
    end

    masks = get_masks(positions, params.dot_radius, params.img_height,
                      params.img_width, params.area_height, params.area_width)

    return init_positions, masks, positions
end
