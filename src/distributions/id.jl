export id_dist,
    IDDist

using Distributions

struct IDDist <: Gen.Distribution{Int64} end

const id_dist = IDDist()

function Gen.random(::IDDist, z::Int64)
    z
end

function Gen.logpdf(::IDDist, x::Int64, z::Int64)
    x == z ? 0. : -Inf
end

(::IDDist)(z) = Gen.random(id_dist, z)

Gen.has_output_grad(::IDDist) = false
Gen.logpdf_grad(::IDDist, value::Set, args...) = (nothing,)
