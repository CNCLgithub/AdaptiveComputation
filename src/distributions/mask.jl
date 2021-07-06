

export mask

"""
Random variable describing a mask prediction.
A matrix of bernoullis parametrized by ps::Matrix{Float64}.
Samples a BitMatrix.
"""
struct Mask <: Gen.Distribution{BitMatrix} end

const mask = Mask()


function Gen.random(::Mask, ps::SubArray)
    Gen.random(mask, sparse(ps))
end
function Gen.random(::Mask, ps::AbstractSparseMatrix{Float64})
    result = falses(size(ps))
    xs, ys, vs = findnz(ps)
    for i = 1:nnz(ps)
        result[xs[i], ys[i]] = bernoulli(vs[i])
    end
    return result
end
function Gen.random(::Mask, ps::Union{Matrix{Float64},
                                      Fill{Float64}})
    result = falses(size(ps))
    for i in eachindex(ps)
        if bernoulli(ps[i])
            result[i] = true
        end
    end
    return result
end

function Gen.logpdf(::Mask, image::BitMatrix, ps::SubArray)
    Gen.logpdf(mask, image, sparse(ps))
end
function Gen.logpdf(::Mask, image::BitMatrix, ps::AbstractSparseMatrix{Float64})
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
function Gen.logpdf(::Mask, image::BitMatrix, ps::Matrix{Float64})
    lpdf = 0.
    for i in eachindex(ps)
        lpdf += Gen.logpdf(bernoulli, image[i], ps[i])
    end
    return lpdf
end
function Gen.logpdf(::Mask, image::BitMatrix, ps::Fill{Float64})
    p = first(ps)
    s = sum(image)
    lpnot = log(1.0 - p)
    log(exp(s * log(p)) + exp((length(image) - s) * lpnot))
end


(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
