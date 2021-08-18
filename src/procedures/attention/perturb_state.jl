export perturb_state!


"""
	state_perturb(trace, probs)

Perturbs velocity based on probs of assignments to observations.
"""

@gen  function state_proposal(trace::Gen.Trace, tracker::Int64,
                                      att::MapSensitivity)
    t, gm, dm, gr = Gen.get_args(trace)
    choices = Gen.get_choices(trace)

    # rng = collect(Int64, max(1, t - steps):t)
    # n = length(rng)

    # ia = choices[ia_addr]

    # ia_addr = :kernel => first(rng) => :dynamics =>
    #     :trackers => tracker => :ang
    # ia = choices[ia_addr]

    # {ia_addr} ~ von_mises(ia, 100.)
    # {:kernel} ~ Gen.Unfold(angle_proposal)(rng, tracker, ia, k)
    i = max(1, t-att.ancestral_steps)
    addr = :kernel => i => :dynamics => :trackers => tracker => :ang
    ang = choices[addr]
    inertia = choices[:kernel => i => :dynamics => :trackers => tracker => :inertia]
    @unpack k_min, k_max = dm
    k = inertia ? k_max : k_min
    {addr} ~ von_mises(ang, k)

    return nothing
end

function apply_random_walk(trace::Gen.Trace, proposal, proposal_args)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    # @show fwd_weight
    (new_trace, weight, _, discard) = update(trace,
        model_args, argdiffs, fwd_choices)
    # @show weight
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = Gen.assess(proposal, proposal_args_backward, discard)
    # @show bwd_weight
    alpha = weight - fwd_weight + bwd_weight
    (new_trace, weight)
end

function tracker_kernel(trace::Gen.Trace, tracker::Int64,
                        att::MapSensitivity)
    new_tr, w1 = apply_random_walk(trace, state_proposal,
                                   (tracker, att))
    # @show w1
    new_tr, w2 = ancestral_tracker_move(new_tr, tracker, att)
    # @show w2
    (new_tr, w1 + w2)
end

# rejuvenate_state!(state, probs) = rejuvenate!(state, probs, state_move)

function ancestral_tracker_move(trace::Gen.Trace, tracker::Int64, att::MapSensitivity)
    args = Gen.get_args(trace)
    t = first(args)
    addrs = []
    t == 1 && return (trace, 0.)

    for i = max(2, t-att.ancestral_steps):t
        addr = :kernel => i => :dynamics => :trackers => tracker
        push!(addrs, addr)
    end
    (new_tr, ll) = take(regenerate(trace, Gen.select(addrs...)), 2)
end

"""
    rejuvenate_state!(state::Gen.ParticleFilterState, probs::Vector{Float64})

    Does one state rejuvenation step based on the probabilities of which object to perturb.
    probs are softmaxed in the process so no need to normalize.
"""
function perturb_state!(state::Gen.ParticleFilterState, att::AbstractAttentionModel,
                        probs::Vector{Float64},
                        sweeps::Int64)
    num_particles = length(state.traces)
    accepted = 0
    attended_trackers = zeros(length(probs))
    for i=1:num_particles
        tracker = Gen.categorical(probs)
        attended_trackers[tracker] += sweeps
        for j = 1:sweeps
            new_tr, ls = att.jitter(state.traces[i], tracker, att)
            if log(rand()) < ls
                state.traces[i] = new_tr
                accepted += 1
            end
        end
    end
    acceptance_rate = (accepted) / (num_particles * sweeps)
    attended_rate = attended_trackers / num_particles
    return (acceptance_rate, attended_rate)
end
