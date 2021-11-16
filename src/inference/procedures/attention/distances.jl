using OptimalTransport
using StatsBase: pairwise

function discrete_measure(d::Dict{K, Float64},
                          scale::Float64) where {K}
    lws = collect(values(d)) .* scale
    lws = lws .- logsumexp(lws)
    ws = exp.(lws)
    ks = collect(keys(d))
    # non_zero = not_approx_zero.(ws)
    # non_zero = lws .> -300
    # ws = ws[non_zero]
    # ks = collect(keys(d))[non_zero]
    (ks, ws)
end

function td_cost(x::Int64, y::Int64)::Float64
    x === y ? 0. : 1.
end

function td_cost(a::BitVector, b::BitVector)::Float64
    aorb = sum(map(|, a, b))
    aorb === 0 && return 0.
    axorb = sum(map(⊻, a, b))
    axorb / aorb
end

function sinkhorn_div(p::Dict{K,V}, q::Dict{K,V};
                      λ::Float64 = 1.0,
                      ε::Float64 = 0.01,
                      scale::Float64 = 1.0) where {K, V}
    a_k, a_w = discrete_measure(p, scale)
    b_k, b_w = discrete_measure(q, scale)
    c = pairwise(td_cost, a_k, b_k)
    ot = sinkhorn_unbalanced(a_w, b_w, c, λ, λ, ε)
    d = sum(ot .* c)
    # display(p)
    # display(q)
    # @show d
    isnan(d) || d < 0. ? 0. : d
end

function sinkhorn_div(ps::Array{Dict{K,V}}, qs::Array{Dict{K,V}}; kwargs...) where {K,V}
    @>> map((p,q) -> sinkhorn_div(p,q;kwargs...), ps, qs) mean
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
        vals[i, 1] = get(p, k, -Inf)
        vals[i, 2] = get(q, k, -Inf)
    end
    # println("probs = ")
    # display(Dict(zip(s, eachrow(vals))))
    vals[:, 1] .-= logsumexp(vals[:, 1])
    vals[:, 2] .-= logsumexp(vals[:, 2])
    display(Dict(zip(s, eachrow(vals))))
    (s, vals)
end

# returns relative entropy (in particular, Jensen-Shannon divergence)
# between p and q distributions
function js_div(p::Dict{K,V}, q::Dict{K,V};
                error_on_empty=true) where {K, V}

    m1 = DiscreteMeasure()
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
        # event is zero prob in both cases
        probs[i, 1] === -Inf && probs[i, 2] === -Inf && continue
        _kl = 0.5 * exp(probs[i, 1]) * (probs[i, 1] - ms[i])
        _kl += 0.5 * exp(probs[i, 2]) * (probs[i, 2] - ms[i])
        kl += _kl
        # _kl = round(_kl, sigdigits = 8)
        # kl += (_kl == Inf) || isnan(_kl) ? 1.0 : _kl # set maximum div
        println("$(labels[i]) => $(probs[i, :]) | $(ms[i]) => $(log(_kl)), $(log(kl))")
    end
    clamp(kl, 0., 1.)
end
# this is for receptive fields
function js_div(ps::Array{Dict{K,V}}, qs::Array{Dict{K,V}}) where {K,V}
    @>> map(js_div, ps, qs) mean
end

const not_approx_zero = !isapprox(0.; atol = 1E-12)



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
