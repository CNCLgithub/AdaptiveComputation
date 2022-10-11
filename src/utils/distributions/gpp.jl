export gpp
export GaussObs, GaussianPointProcess, GaussianComponent


GaussObs{N} = Vector{SVector{N, Float64}}

struct GaussianPointProcess{N} <: Gen.Distribution{GaussObs{N}} end

const gpp = GaussianPointProcess{2}()

struct GaussianComponent{N}
    w::Float64
    mu::SVector{N, Float64}
    cov::SMatrix{N, N, Float64}
end

function Gen.random(::GaussianPointProcess{N},
                    ps::AbstractVector{GaussianComponent{N}}) where {N}
    nc = length(ps)
    result = Vector{SVector{N, Float64}}(undef, nc)
    @inbounds for i = 1:nc
        c = ps[i]
        result[i] = mvnormal(c.mu, c.cov)
    end
    return result
end

function Gen.logpdf(::GaussianPointProcess{N},
                    xs::GaussObs{N},
                    cs::AbstractVector{GaussianComponent{N}}) where {N}
    @assert length(xs) == length(cs)
    n = length(cs)
    # ls = Vector{Float64}(undef, n)
    ls::Float64 = 0.
    @views @inbounds for i = 1:n
        x = xs[i]
        c = cs[i]
        @unpack w, mu, cov = c
        # ls += w
        ls += Gen.logpdf(mvnormal, x, mu, cov)
    end
    # return logsumexp(ls)
    return ls
end

(::GaussianPointProcess{N})(ps) where {N} = Gen.random(GaussianPointProcess{N}(), ps)

Gen.has_output_grad(::GaussianPointProcess) = false
Gen.logpdf_grad(::GaussianPointProcess, value::Set, args...) = (nothing,)
