"""
Computes the entropy of a discrete distribution
"""
function entropy(ps::AbstractArray{Float64})
    # -k_B * sum(map(p -> p * log(p), ps))
    normed = ps .- logsumexp(ps)
    s = 0
    for (p,n) in zip(ps, normed)
        s += p * exp(n)
    end
    -1 * s
end

function entropy(pd::Dict)
    lls = collect(Float64, values(pd))
    log(entropy(lls))
end

"""
    check whether p and q have the same keys
    if there's a change, see how small the val is for the missing
    component and see how big the divergence is
"""
function resolve_correspondence(p::Dict{K,V}, q::Dict{K,V}) where {K, V<:Float64}
    s = collect(union(keys(p), keys(q)))
    vals = Matrix{Float64}(undef, length(s), 2)
    for (i, k) in enumerate(s)
        pv = get(p, k, 0.)
        if pv === 0
            pv = -Inf
        end
        qv = get(q, k, 0.)
        if qv === 0
            qv = -Inf
        end
        vals[i, 1] = pv
        vals[i, 2] = qv
    end
    # println("probs = ")
    # display(Dict(zip(s, eachrow(vals))))
    vals[:, 1] .-= logsumexp(vals[:, 1])
    vals[:, 2] .-= logsumexp(vals[:, 2])
    # display(Dict(zip(s, eachrow(vals))))
    (s, vals)
end

# this is for receptive fields
function js_div(ps::T, qs::T) where T<:Array
    @>> map(js_div, ps, qs) mean
end

# returns relative entropy (in particular, Jensen-Shannon divergence)
# between p and q distributions
function js_div(p::T, q::T;
                error_on_empty=true) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error("empty intersect")
        return 1.0
    end
    ms = Vector{Float64}(undef, size(probs, 1))
    @inbounds for i = 1:size(probs, 1)
        ms[i] = logsumexp(probs[i, :]) - log(2)
    end
    order = sortperm(probs[:, 1], rev= true)
    kl = 0.0
    @inbounds for i in order
        probs[i, 1] === -Inf && probs[i, 2] === -Inf && continue
        _kl = 0.5 * exp(probs[i, 1]) * (probs[i, 1] - ms[i])
        _kl += 0.5 * exp(probs[i, 2]) * (probs[i, 2] - ms[i])
        # _kl = round(_kl, sigdigits = 8)
        kl += (_kl == Inf) || isnan(_kl) ? 1.0 : _kl # set maximum div
        # println("$(labels[i]) => $(probs[i, :]) | $(ms[i]) => $(log(_kl)), $(log(kl))")
    end
    clamp(kl, 0., 1.)
end


function get_entropy(ps::Vector{Float64})
    @>> ps map(p -> (-p * log(p))) sum
end

# average jeffs divergence between ps and qs
function jeffs_d(ps::T, qs::T) where T<:Array
    @>> zip(ps, qs) begin
        map(x -> jeffs_d(x[1], x[2]; error_on_empty=false))
        mean
    end
end

# jeffs divergence between p and q distributions
function jeffs_d(p::T, q::T;
                 error_on_empty=true) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error("empty intersect")
        return 1.0
    end
    # println("probs = ")
    # display(Dict(zip(labels, eachrow(probs))))

    # order = sortperm(probs[:, 1], rev= true)
    n = length(labels)
    jd = 0.0
    @inbounds for i = 1:n
        _jd = exp(probs[i, 1]) - exp(probs[i, 2])
        _jd = _jd == 0. ? 0. : _jd * (probs[i, 1] - probs[i, 2])
        jd += _jd
        # println("$(labels[i]) => $(probs[i, :]) => jd = $(_jd)")
    end

    return jd
end

# average l2 distance between ps and qs distributions
function l2_d(ps::T, qs::T) where T<:Array
    @>> zip(ps, qs) begin
        map(x -> l2_d(x[1], x[2]; error_on_empty=false))
        mean
    end
end

# l2 distance between p and q distributions
function l2_d(p::T, q::T;
                 error_on_empty=true) where T<:Dict
    labels, probs = resolve_correspondence(p, q)
    if isempty(labels)
        display(p); display(q)
        error_on_empty && error("empty intersect")
        return 1.0
    end
    println("probs = ")
    display(Dict(zip(labels, eachrow(probs))))
    # norm(probs[:, 1] - probs[:, 2])
    norm(exp.(probs[:, 1]) - exp.(probs[:, 2]))
end
