
"""
Random variable describing a mask. Essentially just a
bunch of independent bernoullis parametrized by ps::Matrix{Float64}.
Samples a BitMatrix.
"""

export mask

struct Mask <: Gen.Distribution{Array} end

const mask = Mask()


function Gen.random(::Mask, ps::Matrix{Float64})::BitArray
    BitArray(Gen.bernoulli.(ps))
end

b_pdf(x::Bool, p::Float64) = Gen.logpdf(bernoulli, x, p)

function Gen.logpdf(::Mask, image::Matrix, ps::Matrix{Float64})::Float64
    lpdf = sum(b_pdf.(image, ps))
end

(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
