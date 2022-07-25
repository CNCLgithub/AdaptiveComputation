

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
    xs, ys, vs = findnz(ps)
    result = falses(size(ps))
    @inbounds for i = eachindex(vs)
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
    return sparse(result)
end

const min_mask_ls = log(1E-4)

function Gen.logpdf(::Mask,
                    image::BitMatrix,
                    ps::SparseMatrixCSC{Float64})
    # PDF regions:
    #  a - the intersection between `image` and `ps`
    #  b - the non-zero region of `ps`
    #  c - the non-zero region of `image`
    @assert size(image) == size(ps) "weights have size $(size(ps)) but mask has size $(size(image))"

    rows = rowvals(ps)
    vs = nonzeros(ps)
    m, n = size(ps)
    ab = 0.
    c = 0
    @inbounds @views for j = 1:n
        for i in nzrange(ps, j)
            x = image[rows[i], j]
            v = vs[i]
            @fastmath ab += x ? log(v) : log(1.0 - v)
            # @fastmath ab += abs(v - x)
            c += x
        end
    end

    ni = sum(image)
    nz = length(vs)

    lc = abs(ni - c) * min_mask_ls
    lpdf = ab + lc
    # lpdf = -log(ab + abs(ni - c) + 1) * 50.0
    return lpdf
end

function Gen.logpdf(::Mask,
                    image::BitMatrix,
                    ps::Matrix{Float64})
    lpdf = 0.
    @inbounds for i in eachindex(ps)
        lpdf += Gen.logpdf(bernoulli, image[i], ps[i])
    end
    return lpdf
end
function Gen.logpdf(::Mask,
                    image::BitMatrix,
                    ps::Fill{Float64})
    p = first(ps)
    s = sum(image)
    (s * log(p)) + ((length(image) - s) * log(1.0 - p))
end


(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
