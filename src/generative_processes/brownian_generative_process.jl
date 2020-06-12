export brownian_generative_process

function brownian_generative_process(T::Int, params::Dict)
    
    d_params = params["dynamics_params"]
    dynamics_model = BrownianDynamicsModel(d_params["inertia"], d_params["spring"], d_params["sigma_w"])

    num_dots = params["num_dots"]
    dots = Vector{Dot}(undef, num_dots)

    # initial positions and positions over time will be returned
    # from this generative process
    init_positions = Array{Float64}(undef, num_dots, 3)
    positions = Array{Float64}(undef, T, num_dots, 3)


    # randomly initializing the dots
    for i=1:num_dots
        init_pos = params["init_pos_spread"]
        x = uniform(-init_pos, init_pos)
        y = uniform(-init_pos, init_pos)
        z = uniform(0, 1) # depth
        
        init_vel = params["init_vel_spread"]
        vx = uniform(-init_vel, init_vel)
        vy = uniform(-init_vel, init_vel)

        dots[i] = Dot([x,y,z], [vx,vy])
        init_positions[i,:] = dots[i].pos
    end
    

    for t=1:T
        dots = update(dots, dynamics_model)
        for i=1:num_dots
            positions[t,i,:] = dots[i].pos
        end
    end
	
	return init_positions, positions
end
