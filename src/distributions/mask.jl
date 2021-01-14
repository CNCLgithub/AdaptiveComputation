# random variable describing masks
# essentially just a bunch of independent bernoullis

export mask

struct Mask <: Gen.Distribution{Array} end

const mask = Mask()

function Gen.random(::Mask, ps::Matrix{Float64})
    image = BitArray(Gen.bernoulli.(ps))
	return image
end

b_pdf(x::Bool, p::Float64) = Gen.logpdf(bernoulli, x, p)

Gen.logpdf(::Mask, image::Matrix, ps::Matrix{Float64}) = sum(b_pdf.(image, ps))

(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
