
"""
Random variable describing a mask. Essentially just a
bunch of independent bernoullis parametrized by ps::Matrix{Float64}.
Samples a BitMatrix.
"""

export mask

struct Mask <: Gen.Distribution{BitMatrix} end

const mask = Mask()


function Gen.random(::Mask, ps::Matrix{Float64})
    result = BitArray{2}(undef, size(ps))
    for i in eachindex(ps)
        if bernoulli(ps[i])
            result[i] = true
        end
    end
    return result
end

function Gen.logpdf(::Mask, image::BitMatrix, ps::Matrix{Float64})::Float64
    lpdf = 0.
    for i in eachindex(ps)
        lpdf += Gen.logpdf(bernoulli, image[i], ps[i])
    end
    return lpdf
end

(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
