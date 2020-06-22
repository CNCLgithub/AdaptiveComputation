export perturb_state!

"""
	state_perturb(trace, probs)

Perturbs velocity based on probs of assignments to observations.
"""
@gen function state_perturb_proposal(trace, probs, attended_trackers)
    t, motion, gm = Gen.get_args(trace)
    choices = Gen.get_choices(trace)

    # sample a tracker to perturb
    tracker_ps = softmax(probs)
    tracker = @trace(Gen.categorical(tracker_ps), :tracker)
    
    # perturb velocity
    addr_vx = :states => t => :dynamics => :brownian => tracker => :vx
    addr_vy = :states => t => :dynamics => :brownian => tracker => :vy
    prev_v = [choices[addr_vx], choices[addr_vy]]

    # maybe unfair advantage to our new model
    # (was constant at 2.5 before)
    @trace(broadcasted_normal(prev_v, motion.sigma_w), :new_v)

    return tracker, prev_v
end


"backward step for state perturbation"
function state_perturb_involution(trace, fwd_choices::ChoiceMap, fwd_ret,
                                   proposal_args::Tuple)
    choices = Gen.get_choices(trace)
    t, motion, gm = Gen.get_args(trace)
    (tracker, prev_v) = fwd_ret

    # recording attended tracker in involution
    # (not to count twice)
    attended_trackers = proposal_args[2]
    attended_trackers[tracker] += 1

    # constraints for update step
    constraints = choicemap()

    # decision over target state
    vx, vy = fwd_choices[:new_v]
    #constraints[:states => t => :trackers => tracker => :vx] = vx
    #constraints[:states => t => :trackers => tracker => :vy] = vy
    constraints[:states => t => :dynamics => :brownian => tracker => :vx] = vx
    constraints[:states => t => :dynamics => :brownian => tracker => :vy] = vy

    # backward stuffs
    bwd_choices = choicemap()
    bwd_choices[:tracker] = fwd_choices[:tracker]
    bwd_choices[:new_v] = prev_v

    model_args = get_args(trace)
    (new_trace, weight, _, _) = Gen.update(trace, model_args, (NoChange(),), constraints)

    (new_trace, bwd_choices, weight)
end

state_move(trace, args) = Gen.mh(trace, state_perturb_proposal, args, state_perturb_involution)

rejuvenate_state!(state, probs) = rejuvenate!(state, probs, state_move)


"""
    rejuvenate_state!(state::Gen.ParticleFilterState, probs::Vector{Float64})

    Does one state rejuvenation step based on the probabilities of which object to perturb.
    probs are softmaxed in the process so no need to normalize.
"""
function perturb_state!(state::Gen.ParticleFilterState, probs::Vector{Float64})
    #timestep, motion, gm = Gen.get_args(first(state.traces))
    num_particles = length(state.traces)
    accepted = 0
    attended_trackers = zeros(length(probs))

    args = (probs, attended_trackers)
    for i=1:num_particles
        state.traces[i], a = state_move(state.traces[i], args)
        accepted += a
    end
    
    return accepted/num_particles, attended_trackers/num_particles
end
