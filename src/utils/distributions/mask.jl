import IfElse
using SparseArrays: getcolptr

export mask

"""
Random variable describing a mask prediction.
A matrix of bernoullis parametrized by ps::Matrix{Float64}.
Samples a BitMatrix.
"""
struct Mask <: Gen.Distribution{Matrix{Bool}} end

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

my_bern(w) = rand() < w

function Gen.random(::Mask, ps::Union{Matrix{Float64},
                                      Fill{Float64}})
    result = Matrix{Bool}(undef, size(ps))
    @inbounds for i in eachindex(ps)
        result[i] = bernoulli(ps[i])
    end
    return result
end

const min_mask_ls = log(1E-4)
const comp_mmls = log(1.0 - 1E-4)


function Gen.logpdf(::Mask,
                    image::Matrix{Bool},
                    ps::SparseMatrixCSC{Float64})
    # PDF regions:
    #  a - the intersection between `image` and `ps`
    #  b - the non-zero region of `ps`
    #  c - the non-zero region of `image`
    # @assert size(image) == size(ps) "weights have size $(size(ps)) but mask has size $(size(image))"
    rows = rowvals(ps)
    vs = nonzeros(ps)
    ncol = size(ps, 2)
    a = 0.
    k = 0 # number of `on` pixels encountered in `vs`
    @inbounds @views @fastmath for j in 1:ncol
        for i in nzrange(ps, j)
            x = image[rows[i], j]
            v = vs[i]
            a += x ? log(v) : log(1.0-v)
            k += x
        end
    end
    s = sum(image) # total number of on pixels
    # some `on` pixels were not covered by `vs`
    b = s == k ? 0.0 : (s - k) * min_mask_ls
    n = length(image)
    z = length(vs)
    # account for `off` pixels outside of `vs`
    c = n == s ? 0.0 : (n - s - (z - k)) * comp_mmls
    # @show a
    # @show b
    # @show c
    lpdf = a + b + c
    # @show lpdf
    return lpdf
end

function Gen.logpdf(::Mask,
                    image::Matrix{Bool},
                    ps::Matrix{Float64})
    # vmapreduce(bern_vectorized, +, ps, image)
    lpdf = 0.
    @inbounds @views @fastmath for i in indices((ps, image))
        # lpdf += bern_vectorized(ps[i], image[i])
        lpdf += IfElse.ifelse(image[i], log(ps[i]), log(1 - ps[i]))
    end
    return lpdf
end
function Gen.logpdf(::Mask,
                    image::Matrix{Bool},
                    ps::Fill{Float64})
    p = first(ps)
    s = sum(image)
    li = length(image)
    (s * log(p)) + ((li - s) * log(1.0 - p))
    # logsumexp(log(s) + log(p), log(li - s) + log(1.0 - p)) - log(li)
end


(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
