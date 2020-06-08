# random variable describing masks
# essentially just a bunch of independent bernoullis

export mask

struct Mask <: Gen.Distribution{Array} end

const mask = Mask()

function Gen.random(::Mask, ps::Matrix{Float64})
    image = Gen.bernoulli.(ps)
	return image
end

function Gen.logpdf(::Union{Mask, Nothing}, image::BitArray{2}, ps::Matrix{Float64})
    bernoullis = fill(bernoulli, size(ps))
    lpdfs = Gen.logpdf.(bernoullis, image, ps)
    return sum(lpdfs)
end


(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)

