export Dot

mutable struct Dot <: Object
    pos::Vector{Float64}
    vel::Vector{Float64}
end



@gen function data_generating_procedure(state::Union{Nothing, State}, t::Int, params::Params)
	
	dots = Vector{Dot}(undef, params.num_observations)
	init_dots = Array{Float64}(undef, params.num_observations, 2)

	new_state = State(dots)
	
	for d=1:params.num_observations
		if typeof(state) == Nothing
			x = @trace(uniform(-250.0, 250.0), (:x, 1, d))
			y = @trace(uniform(-250.0, 250.0), (:y, 1, d))
			init_dots[d,1] = x
			init_dots[d,2] = y
			vx = @trace(normal(0.0, params.sigma_v), (:vx, 1, d))
			vy = @trace(normal(0.0, params.sigma_v), (:vy, 1, d))
		else
			x = state.dots[d].x
			y = state.dots[d].y
			vx = state.dots[d].vx
			vy = state.dots[d].vy

			vx = @trace(normal(params.inertia * vx - params.spring * x, params.sigma_w), (:vx, t, d))
			vy = @trace(normal(params.inertia * vy - params.spring * y, params.sigma_w), (:vy, t, d))
		end

		x += vx
		y += vy
		
		cov = [params.measurement_noise, params.measurement_noise]
		@trace(Gen.broadcasted_normal([x,y], cov), (:obs, t, d))

		new_state.dots[d] = Dot(x, y, vx, vy)
	end	
	
	return new_state, init_dots
end


# runs data generating procedure
# returns observations, positions of the dots, average velocity of the dots
function collect_observations(gen_fn, T::Int, params::Params)
	current_state = nothing

	obs = Array{Float64}(undef, T, params.num_observations, 2)
	dots = Array{Float64}(undef, T, params.num_observations, 2)
	# the very initial state before moving
	init_dots = Array{Float64}(undef, params.num_observations, 2)
	avg_vel_sum = 0.0

	for t=1:T
		current_trace = Gen.simulate(gen_fn,
									 (current_state, t, params,))
		if t==1
			current_state, init_dots = Gen.get_retval(current_trace)
		else
			current_state, _ = Gen.get_retval(current_trace)
		end

		choices = Gen.get_choices(current_trace)
		for	i=1:params.num_observations
			obs[t,i,:] = choices[(:obs, t, i)]
		end
		
		dots_temp = current_state.dots
		for dot in dots_temp
			avg_vel_sum += abs(dot.vx) + abs(dot.vy)
		end

		for	i=1:params.num_observations
			dots[t,i,:] = [dots_temp[i].x, dots_temp[i].y]
		end
	end
	
	return obs, avg_vel_sum/params.num_observations/T, dots, init_dots
end


