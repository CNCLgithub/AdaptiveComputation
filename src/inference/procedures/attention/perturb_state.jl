export tracker_kernel

@gen function state_proposal(trace::Gen.Trace, latent::Int, k::Int)

    ct = first(get_args(trace))
    t = max(1, ct - k)
    # first sample new inertia or collision event
    inr_addr = :kernel => t => :trackers => latent => :inertia
    inr = {inr_addr} ~ bernoulli(0.50)

    # then update angle of motion
    ang_addr = :kernel => t => :trackers => latent => :ang
    prev_ang = trace[ang_addr]
    ang = {ang_addr} ~ von_mises(prev_ang, 150.0)

    # finally update motion speed
    mag_addr = :kernel => t => :trackers => latent => :mag
    prev_mag = trace[mag_addr]
    mag = {mag_addr} ~ normal(prev_mag, 0.5)
    result = (inr, ang, mag)
    return result
end

function apply_random_walk(trace::Gen.Trace, proposal, proposal_args)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = Gen.update(trace,
                                                 fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = Gen.assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    if isnan(alpha)
        @show fwd_weight
        @show weight
        @show bwd_weight
        display(discard)
        error("nan in proposal")
    end
    (new_trace, alpha)
end

function regenerate_trajectory(trace::Gen.Trace, tracker::Int, k::Int)
    ct = first(get_args(trace))
    t = max(1, ct - k)
    t == 1 && return (trace, 0.)
    addrs = []
    for i = (t+1):ct
        push!(addrs,
              :kernel => i => :trackers => tracker)
    end
    (new_tr, ll, _) = regenerate(trace, Gen.select(addrs...))
end

function tracker_kernel(trace::Gen.Trace, tracker::Int, t::Int)
    new_tr, w1 = apply_random_walk(trace, state_proposal, (tracker, t))
    new_tr, w2 = regenerate_trajectory(new_tr, tracker, t)
    (new_tr, w1 + w2)
end
