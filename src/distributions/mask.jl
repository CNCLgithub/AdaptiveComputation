# random variable describing masks
# essentially just a bunch of independent bernoullis

export mask

struct Mask <: Gen.Distribution{Array} end

const mask = Mask()

function Gen.random(::Mask, ps::Matrix{Float64})
    image = BitArray(Gen.bernoulli.(ps))
	return image
end

function Gen.logpdf(::Mask, image::Matrix, ps::Matrix{Float64})
    ll = 0
    for i in eachindex(image)
        ll += Gen.logpdf(bernoulli, image[i], ps[i])
    end
    ll
end


(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)

