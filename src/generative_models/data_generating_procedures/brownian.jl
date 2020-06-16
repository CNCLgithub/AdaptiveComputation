export dgp

function dgp(k::Int, params::GMMaskParams,
             motion::BrownianDynamicsModel)

    init_state, states = gm_masks_static(k, motion, params)

    num_dots = params.n_trackers
    dots = Vector{Dot}(undef, num_dots)

    # initial positions and positions over time will be returned
    # from this generative process
    init_positions = Array{Float64}(undef, num_dots, 3)
    positions = Array{Float64}(undef, k, num_dots, 3)

    for i=1:size(init_state.graph.elements)
        init_positions[i,:] = init_state.graph.elements.pos
    end
    
    for t=1:k
        dots = states[t].graph.elements
        for i=1:num_dots
            positions[t,i,:] = dots[i].pos
        end
    end

    masks = get_masks(positions, params)
    return init_positions, masks
end
