export perturb_state!

addr_from_base(base::Nothing, addr) = (addr,)
addr_from_base(base::Tuple, addr) = (base..., addr)
function select_from_cm!(s::Selection, base::Tuple, cm::ChoiceMap)
    for (addr, value) in get_values_shallow(cm)
        dst = addr_from_base(base, addr)
        push!(s, foldr(Pair, dst))
    end
    for (addr, submap) in get_submaps_shallow(cm)
        dst = addr_from_base(base, addr)
        select_from_cm!(s, dst, submap)
    end
    return nothing
end
function select_from_cm(cm::ChoiceMap)
    s = Gen.select()
    select_from_cm!(s, (), cm)
    s
end

function verify_update(tr::Gen.Trace, fwd::ChoiceMap)
    # first eval fwd choice
    s = select_from_cm(fwd)
    display(fwd)
    fwd_ls = project(tr, s)
    println("Fwd choice: $(fwd_ls)")

    cm = get_choices(tr)
    s = select_from_cm(cm)
    println("total ls: $(project(tr, s))")

    t = first(get_args(tr))
    s = Gen.select(:kernel => t => :masks)
    ls = project(tr, s)
    println("P(x | h): $(ls)")

    cm_prior = get_selected(cm, Gen.complement(s))
    s = select_from_cm(cm_prior)
    ls = project(tr, s)
    println("P(h): $(ls)")
    println("Score of trace $(get_score(tr))")
    return nothing
end

function _state_proposal(trace::Gen.Trace, tracker::Tuple,
                         att::MapSensitivity)
    aaddr = foldr(Pair, (tracker..., :ang))
    maddr = foldr(Pair, (tracker..., :mag))
    iaddr = foldr(Pair, (tracker..., :inertia))
    (aaddr, maddr, iaddr)
end

# TODO: perturb initial state
@gen  function state_proposal(trace::Gen.Trace, tracker::Tuple,
                              att::MapSensitivity)
    (aaddr, maddr, iaddr) = _state_proposal(trace, tracker, att)
    {iaddr} ~ bernoulli(0.50)
    ang = trace[aaddr]
    {aaddr} ~ von_mises(ang, 150.0)
    mag = trace[maddr]
    {maddr} ~ normal(mag, 0.1)
    return nothing
end

function apply_random_walk(trace::Gen.Trace, proposal, proposal_args)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = Gen.update(trace,
        model_args, argdiffs, fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = Gen.assess(proposal, proposal_args_backward, discard)
    alpha = get_score(trace) === -Inf ? 1.0 : weight - fwd_weight + bwd_weight
    if isnan(alpha)
        @show fwd_weight
        @show weight
        @show bwd_weight
        display(discard)
        verify_update(trace, fwd_choices)
        verify_update(new_trace, fwd_choices)
        error("nan in proposal")
    end
    (new_trace, alpha)
end

function tracker_kernel(trace::Gen.Trace, tracker::Tuple,
                        att::MapSensitivity)
    # first update inertia
    # new_tr, w1 = ancestral_inertia_move(trace, tracker, att)
    new_tr, w2 = apply_random_walk(trace, state_proposal,
                                   (tracker, att))
    # (new_tr, w1 + w2)
    (new_tr, w2)
end

# rejuvenate_state!(state, probs) = rejuvenate!(state, probs, state_move)

function ancestral_inertia_move(trace::Gen.Trace, tracker::Tuple, att::MapSensitivity)
    args = Gen.get_args(trace)
    iaddr = foldr(Pair, (tracker..., :inertia))
    addrs = []
    (@> trace get_choices has_value(iaddr)) && push!(addrs, iaddr)
    isempty(addrs) && return (trace, 0.)
    (new_tr, ll) = take(regenerate(trace, Gen.select(addrs...)), 2)
end

"""
    rejuvenate_state!(state::Gen.ParticleFilterState, probs::Vector{Float64})

    Does one state rejuvenation step based on the probabilities of which object to perturb.
    probs are softmaxed in the process so no need to normalize.
"""
function perturb_state!(chain::SeqPFChain,
                        att::MapSensitivity)
    @unpack state, auxillary = chain
    @unpack weights, cycles = auxillary
    allocated = Dict{Int64, Vector{Int64}}()
    if sum(cycles) == 0.
        @pack! auxillary = allocated
        return nothing
    end
    num_particles = length(state.traces)
    @views @inbounds for i=1:num_particles
        # skip if -Inf
        get_score(state.traces[i]) === -Inf && continue
        cycles[i] == 0 && continue
        # map obs weights back to target centric weights
        # c = correspondence(state.traces[i])
        # ne = size(c, 2)
        pweights = weights[i]
        nes = length(pweights)
        tracker_addrs = trackers(state.traces[i]) # address for each tracker
        palloc = zeros(Int64, length(pweights))
        for ti = 1:nes
            # tw = sum(c[:, ti] .* weights) # tracker goal relevance
            tw = pweights[ti] # tracker goal relevance
            steps = round(Int64, tw * cycles[i])
            palloc[ti] += steps
            for s = 1:steps
                # perform an mh move
                new_tr, ls = att.jitter(state.traces[i], tracker_addrs[ti], att)
                if log(rand()) < ls
                    # c = correspondence(state.traces[i])
                    # tw = sum(c[:, ti] .* weights)
                    state.traces[i] = new_tr
                end
            end
        end
        allocated[i] = palloc
    end
    @pack! auxillary = allocated
    @pack! chain = state
    return nothing
end
