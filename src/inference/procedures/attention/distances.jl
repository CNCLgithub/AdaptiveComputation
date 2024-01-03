using OptimalTransport
using StatsBase: pairwise

function discrete_measure(d::Dict{K, Float64},
                          scale::Float64) where {K}
    n = length(d)
    lws = Vector{Float64}(undef, n)
    ks = Vector{K}(undef, n)
    i = 1
    for (k, w) in d
        ks[i] = k
        lws[i] = w * scale
        i += 1
    end
    lws .+= -logsumexp(lws)
    (ks, exp.(lws))
end

function discrete_measure(lws::Vector{Float64},
                          scale::Float64)
    exp.(lws)
    # exp.(lws .- logsumexp(lws))
    # softmax(lws; t = scale)
end

function td_cost(x::Int64, y::Int64)::Float64
    x === y ? 0. : 1.
end

function cost_matrix(n::Int)
    m = ones((n, n))
    m[I(n)] .= 0.
    return m
end

function td_cost(a::BitVector, b::BitVector)::Float64
    aorb = sum(map(|, a, b))
    aorb === 0 && return 0.
    axorb = sum(map(⊻, a, b))
    axorb / aorb
end

function sinkhorn_div(p::Dict{K,V}, q::Dict{K,V};
                      λ::Float64 = 1.0,
                      ε::Float64 = 0.1,
                      scale::Float64 = 1.0) where {K, V}
    a_k, a_w = discrete_measure(p, scale)
    b_k, b_w = discrete_measure(q, scale)
    c = pairwise(td_cost, a_k, b_k)
    ot = sinkhorn(a_w, b_w, c, ε;
                  atol = 1E-4,
                  maxiter=10_000)
    d = OptimalTransport.sinkhorn_cost_from_plan(ot, c, ε;
                                                 regularization=false)
    d = log(d)
    isnan(d)  ? -Inf : d
end


function sinkhorn_div(p::Vector{V}, q::Vector{V};
                      eps::Float64 = 1E-5,
                      scale::Float64 = 1.0) where {V}
    d = -Inf
    for i = eachindex(p)
        delta = logdiffexp(p[i],q[i])
        d = logsumexp(d, delta)
    end
    # if d > -40
    #     display(Dict(:u => p, :v => q, :d => d))
    #     for i = eachindex(p)
    #         a = round(p[i]; sigdigits = 5)
    #         b = round(q[i]; sigdigits = 5)
    #         delta = logdiffexp(a,b)
    #         @show (a, b, delta)
    #     end
    # end
    return d
end

function sinkhorn_div(ps::Array{Dict{K,V}}, qs::Array{Dict{K,V}}; kwargs...) where {K,V}
    @>> map((p,q) -> sinkhorn_div(p,q;kwargs...), ps, qs) mean
end
