export npp_proposal

function _prepare_npp(trace, obs)
	timestep, params = Gen.get_args(trace)
	timestep += 1
	#npp_obs = Array{Float64}(undef, 1, params.num_observations, 2)
	npp_obs = Array{Float64}(undef, 1, size(obs, 1), 2)
	npp_pred = Array{Float64}(undef, 1, params.num_targets, 2)

	# if the first timestep, then extract target locations from initial state
	if timestep == 1
		targets = []
		for i=1:params.num_targets
			x = trace[:initial_state => i => :x]
			y = trace[:initial_state => i => :y]
			push!(targets, Target(x, y, 0.0, 0.0, false))
		end
	else
		targets = Gen.get_retval(trace)[end]
	end

	for t=1:params.num_targets
		npp_pred[1,t,:] = [targets[t].x, targets[t].y]
	end

	npp_obs[1,:,:] = obs

	return npp_pred, npp_obs
end

# setting the noisy locations of targets according to observations
@gen function npp_proposal(trace, obs)
	gen_fn = Gen.get_gen_fn(trace)
	timestep, params = Gen.get_args(trace)
	timestep += 1
	choices = Gen.get_choices(trace)

	obs_array = retrieve_obs(obs, timestep)
	num_obs = size(obs_array, 1)
	
	args = (timestep, params)

	(prediction, weight) = Gen.update(trace, args, (UnknownChange(),),
									  Gen.choicemap())

	npp_pred, npp_obs = _prepare_npp(prediction, obs_array)
	targets = @trace(npp(npp_pred, npp_obs), :chain => timestep => :targets)

	leftover = setdiff(collect(1:num_obs), targets)
	distractors = @trace(assignment(length(leftover), leftover),
						 :chain => timestep => :distractors)

	for i=1:min(length(distractors), num_obs)
		o = distractors[i]
		cov = [params.measurement_noise, params.measurement_noise]
		addr = :chain => timestep => :distractor_map => i => :point
		@trace(broadcasted_normal(obs_array[o,:], cov), addr)
	end
end


@gen function masks_to_state_proposal(trace, points)
    t, params = Gen.get_args(trace)

    # proposing E
    # masks_to_E_proposal(trace)
    if !isnothing(points)
        E = @trace(broadcasted_normal(points[t,:,:], noise), :E)
    else
        error("not implemented")
    end
    
    # proposing S from E
    # state_proposal(trace, E) 
end

# function proposing state from
@gen function state_proposal(trace, E)
    error("not implemented")
end
