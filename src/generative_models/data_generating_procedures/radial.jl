export dgp

using Setfield

function dgp(k::Int, params::GMParams,
             motion::RadialMotion)
    
    # new params with all dots
    # having state for data generation
    gm = deepcopy(params)
    gm = @set gm.n_trackers = round(Int, gm.n_trackers + gm.distractor_rate)
    gm = @set gm.distractor_rate = 0.0

    init_state, states = gm_masks_static(k, motion, gm)

    num_dots = gm.n_trackers
    dots = Vector{Dot}(undef, num_dots)

    # initial positions and positions over time will be returned
    # from this generative process
    init_positions = Array{Float64}(undef, num_dots, 3)
    init_vels = Array{Float64}(undef, num_dots, 2)
    positions = Array{Float64}(undef, k, num_dots, 3)

    for i=1:num_dots
        init_positions[i,:] = init_state.graph.elements[i].pos
        init_vels[i,:] = init_state.graph.elements[i].vel
    end
    
    for t=1:k
        dots = states[t].graph.elements
        for i=1:num_dots
            positions[t,i,:] = dots[i].pos
        end
    end

    masks = get_masks(positions, params.dot_radius, params.img_height,
                      params.img_width, params.area_height, params.area_width)

    return init_positions, init_vels, masks
end
