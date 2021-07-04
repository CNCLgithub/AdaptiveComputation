
"""
Random variable describing a mask. Essentially just a
bunch of independent bernoullis parametrized by ps::Matrix{Float64}.
Samples a BitMatrix.
"""

export mask

struct Mask <: Gen.Distribution{BitMatrix} end

const mask = Mask()


function Gen.random(::Mask, ps::Union{SubArray, AbstractSparseMatrix{Float64}})
    result = falses(size(ps))
    xs, ys, vs = findnz(ps)
    for i = 1:nnz(ps)
        result[xs[i], ys[i]] = bernoulli(vs[i])
    end
    return result
end

function Gen.random(::Mask, ps::Matrix{Float64})
    result = falses(size(ps))
    for i in eachindex(ps)
        if bernoulli(ps[i])
            result[i] = true
        end
    end
    return result
end

function Gen.logpdf(::Mask, image::BitMatrix, ps::Union{SubArray,
                                                        AbstractSparseMatrix{Float64}})::Float64
    mag = sum(image)
    # number of heads is impossible given number of non-zero weights
    nnz(ps) < mag && return -Inf
    xs, ys, vs = findnz(ps)
    lpdf = 0.
    count = 0
    for i = 1:nnz(ps)
        x = image[xs[i], ys[i]]
        lpdf += Gen.logpdf(bernoulli, x, vs[i])
        count += x
    end
    # some zero-weight cells contained heads
    count != mag && return -Inf
    lpdf
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
