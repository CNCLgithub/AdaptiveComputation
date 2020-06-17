export rejuvenate_state!,
    retrieve_confusability,
    early_stopping_confusability

"""
for now returns false, no early stopping
"""
function early_stopping_confusability(new_conf, prev_conf)
    return false
end

"""
	retrieve_td_confusability(state::Gen.ParticleFilterState, unweighted::Bool = true)

Returns tracker designation confusability per unique optics element.
"""
function retrieve_confusability(state::Gen.ParticleFilterState, unweighted::Bool = true)
    num_particles = length(state.traces)

    if unweighted
        samples = sample_unweighted_traces(state, num_particles)
    else
        samples = state.traces
    end

    t, motion, gm = Gen.get_args(first(samples))
    #confusability = zeros(length(first(samples)[:states => t => :optics]))

    # let's do confusability in terms of trackers!!!
    confusability = fill(-Inf, gm.n_trackers)

    for i=1:num_particles
        #optics = samples[i][:states => t => :optics]
        masks = samples[i][:states => t => :masks]
        
        # getting tracker designation, assignment and the weights for TD
        # from saved state in pmbrfs_params (Gen hack)
        pmbrfs_stats = Gen.get_retval(samples[i])[2][t].pmbrfs_params.pmbrfs_stats
        tds, As, td_weights = pmbrfs_stats.partitions, pmbrfs_stats.assignments, pmbrfs_stats.ll
        
        # saving main TD and assignment hypothesis
        main_td = tds[1]
        main_A = As[1]

        # comparing them to the other hypotheses
        for j=2:length(tds)

            # these are in the main hypothesis, but not in the alternative
            differing_obs = setdiff(main_td, tds[j])
           
            # finding the trackers that are involved
            # i.e. trackers having differing observations different hypotheses
            # (this part was/is tricky)
            differing_obs_indices = []
            for ob in differing_obs
                push!(differing_obs_indices, findall(x->x==ob, main_td)[1])
            end

            trackers =[]
            for index in differing_obs_indices
                push!(trackers, main_A[index])
            end
    
            #println("differing_optics: $differing_optics")
            #println("differing_optics_indices: $differing_optics_indices")
            #println("trackers $trackers")

            
            for tracker in trackers
                if unweighted
                    confusability[tracker] = logsumexp(confusability[tracker], td_weights[j])
                else
                    # implement weighted confusability if needed
                    # simply multiply by normalized particle weight
                    error("not implemented")
                end
            end
        end

    end
    
    # dividing confusability by num_particles to normalize
    confusability .-= log(num_particles)

    return confusability
end


"Finds all elements of a in b"
findin(a, b) = map(x -> findfirst(y -> y == x, b), a)



"""
	retrieve_preds(trace, tracker_idx)

Retrieves the predicted target state for the given target.
"""
function retrieve_preds(trace, tracker_idx)
   t, _ = Gen.get_args(trace)
   if t == 1
        x = trace[:initial_state => tracker_idx => :x]
        y = trace[:initial_state => tracker_idx => :y]
    else
        targets = Gen.get_retval(trace)[end]
        x,y = targets[tracker_idx].x, targets[tracker_idx].y
    end
    result = Array{Float64, 3}(undef, 1, 1, 2)
    result[1,1,:] = [x,y]
    return result
end

function retrieve_vels(trace, tracker_idx)
    t, _ = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    vx = choices[:chain => t => :target_map => tracker_idx => :vx]
    vy = choices[:chain => t => :target_map => tracker_idx => :vy]
    return [vx, vy]
end



"""
	state_perturb(trace, confusability)

Perturbs velocity based on confusability of assignments to observations.
"""
@gen function state_perturb_proposal_old(trace, td_confusability)
    t, params = Gen.get_args(trace)
    choices = Gen.get_choices(trace)

    # sample a tracker to perturb
    tracker_ps = softmax(td_confusability)
    tracker = @trace(Gen.categorical(tracker_ps), :tracker)
    
    # perturb velocity
    addr_vx = :states => t => :trackers => tracker => :vx
    addr_vy = :states => t => :trackers => tracker => :vy
    prev_v = [trace[addr_vx], trace[addr_vy]]

    # maybe unfair advantage to our new model
    # (was constant at 2.5 before)
    @trace(broadcasted_normal(prev_v, params["dynamics_params"]["sigma_w"]), :new_v)

    return tracker, prev_v
end


"backward step for state perturbation"
function state_perturb_involution_old(trace, fwd_choices::ChoiceMap, fwd_ret,
                                   proposal_args::Tuple)
    choices = Gen.get_choices(trace)
    t, params = Gen.get_args(trace)
    (tracker, prev_v) = fwd_ret
    
    # recording attended tracker in involution
    # (not to count twice)
    push!(params["attended_trackers"][t], tracker)

    # constraints for update step
    constraints = choicemap()

    # decision over target state
    vx, vy = fwd_choices[:new_v]
    constraints[:states => t => :trackers => tracker => :vx] = vx
    constraints[:states => t => :trackers => tracker => :vy] = vy

    # backward stuffs
    bwd_choices = choicemap()
    bwd_choices[:tracker] = fwd_choices[:tracker]
    bwd_choices[:new_v] = prev_v

    model_args = get_args(trace)
    (new_trace, weight, _, _) = update(trace, model_args, (NoChange(),), constraints)

    (new_trace, bwd_choices, weight)
end


state_move_old(trace, args) = Gen.mh(trace, state_perturb,
                                 args, state_involution)

@gen function state_perturb_proposal(trace, td_confusability)
    println("proposal start")
    choices = get_choices(trace)
    t, params = get_args(trace)
    
    ps = softmax(td_confusability)
    ps = [1.0, 0.0, 0.0, 0.0]
    #println("ps $ps")
    tracker = ({:tracker} ~ Gen.categorical(ps))

   
    addr = :states => t => :trackers => tracker => :vx
    prev_vx = choices[addr]
    #println("$prev_vx")
    new_vx = @trace(normal(prev_vx, 2.5), :new_vx)

    println("proposal end")
    println()

    return prev_vx
    
    """
    addr = :states => t => :trackers => tracker => :vy
    prev_vy = choices[addr]
    @trace(normal(prev_vy, 2.5), addr)
    """
end

@involution function state_perturb_involution(model_args, proposal_args, proposal_retval)
    println("involution start")
    t, params = model_args
    prev_vx = proposal_retval
    
    tracker = @read_discrete_from_proposal(:tracker)
    new_vx = @read_continuous_from_proposal(:new_vx)

    addr = :states => t => :trackers => tracker => :vx
    #@write_continuous_to_model(addr, new_vx)

    @write_discrete_to_proposal(:tracker, tracker)
    @write_continuous_to_proposal(:new_vx, prev_vx)

    println("involution end")
    println()
end

state_move(trace, args) = Gen.mh(trace, state_perturb_proposal_old, args, state_perturb_involution_old)

rejuvenate_state!(state, confusability) = rejuvenate!(state, confusability, state_move)


"""
	rejuvenate!(state::Gen.ParticleFilterState, confusability::Vector{Float64}, move)

Does one rejuvenation step based on the particular move (state_move or assignment_move).
"""
function rejuvenate!(state::Gen.ParticleFilterState, td_confusability::Vector{Float64}, move)
    timestep, params = Gen.get_args(first(state.traces))
    num_particles = length(state.traces)
    accepted = 0
    args = (td_confusability,)
    for i=1:num_particles
        state.traces[i], a = move(state.traces[i], args)
        accepted += a
    end
    return accepted / num_particles
end
