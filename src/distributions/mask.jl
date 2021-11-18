

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
    @assert size(image) == size(ps) "weights have size $(size(ps)) but mask has size $(size(image))"
    ni = sum(image)
    nz = nnz(ps)
    # number of heads is impossible given number of non-zero weights
    nz < ni && return -Inf
    xs, ys, vs = findnz(ps)
    lpdf = 0.
    # count = floor(Int64, 0.05*nz) # adding one pixel for numerical buffer
    count = 0
    @views @inbounds for i = 1:nz
        x = image[xs[i], ys[i]]
        lpdf += Gen.logpdf(bernoulli, x, vs[i])
        count += x
    end
    # lpdf += (abs(ni - count) * log(0.01))
    # some zero-weight cells contained heads
    lpdf = count < ni ? -Inf : lpdf

    # if lpdf != -Inf
    # if lpdf == -Inf
    #     @show ni
    #     @show count
    #     println(UnicodePlots.spy(ps))
    #     println(UnicodePlots.spy(image))
    #     # error()
    # end
    # lpdf
end
function Gen.logpdf(::Mask, image::BitMatrix, ps::Matrix{Float64})
    lpdf = 0.
    @inbounds for i in eachindex(ps)
        lpdf += Gen.logpdf(bernoulli, image[i], ps[i])
    end
    return lpdf
end
function Gen.logpdf(::Mask, image::BitMatrix, ps::Fill{Float64})
    p = first(ps)
    s = sum(image)
    (s * log(p)) + ((length(image) - s) * log(1.0 - p))
end


(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
