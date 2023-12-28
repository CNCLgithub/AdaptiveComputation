export von_mises,
    VonMises

using Distributions

struct VonMises <: Gen.Distribution{Float64} end

const von_mises = VonMises()

function Gen.random(::VonMises, mu::Float64, k::Float64)
    d = Distributions.VonMises(mu, k)
    rand(d)
end

const twopi = 2 * pi

function Gen.logpdf(::VonMises, x::Float64, mu::Float64, k::Float64)
    # from https://stackoverflow.com/a/24234924
    x = x - mu
    x = x - twopi * floor((x + pi) / twopi)
    # @show x
    d = Distributions.VonMises(k)
    Distributions.logpdf(d, x)
end

(::VonMises)(mu, k) = Gen.random(von_mises, mu, k)

Gen.has_output_grad(::VonMises) = false
Gen.logpdf_grad(::VonMises, value::Set, args...) = (nothing,)
