using PhysicalConstants


function target_designation(tr::Gen.trace)
end

"""
Computes the entropy of a discrete distribution
"""
function entropy(ps::AbstractArray{Float64})
    -k_B * sum(map(p -> p * log(p), ps))
end

"""
Returns a weighted vector of approximated
elbo derivatives per object
"""
function elbo(s, ss, logs)
    weights = logsumexp(logs)
    ss .* weights - s
end

function zero_order(state::Gen.ParticleFilterState, objective, args, cm)
    map_tr = get_map(state)
    current_s = (entropy ∘ objective)(map_tr)
    perturbations = map(i -> perturb_state(map_tr, i), 1:N)
    trs, lgs = zip(perturbations...)
    entropies = map(entropy ∘ objective, trs)
    elbo(base_h, entropies, lgs)
end

function first_order(state::Gen.ParticleFilterState, objective, args, cm)
    map_tr = get_map(state)
    current_s = (entropy ∘ objective)(map_tr)
    prediction, p_ls = Gen.update(map_tr, args, (UnknownChange,), cm)
    base_h = (entropy ∘ target_designation)(prediction)
    perturbations = map(i -> perturb_state(prediction, i), 1:N)
    trs, lgs = zip(perturbations...)
    entropies = map(entropy ∘ target_designation, trs)
    elbo(base_h, entropies, lgs)
end
