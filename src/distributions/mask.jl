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

using UnicodePlots
function Gen.logpdf(::Mask, image::Matrix, ps::Matrix{Float64})
    lpdf = sum(b_pdf.(image, ps))
    
    @show lpdf
    mat = reverse(ps, dims = 1)
    println(UnicodePlots.heatmap(mat,
                    title = "ps",
                    border = :none,
                    colorbar_border = :none,
                    colormap = :inferno
                    ))
    mat = reverse(image, dims = 1)
    println(UnicodePlots.heatmap(mat,
                    title = "image",
                    border = :none,
                    colorbar_border = :none,
                    colormap = :inferno
                    ))

    return lpdf
end

(::Mask)(ps) = Gen.random(Mask(), ps)

Gen.has_output_grad(::Mask) = false
Gen.logpdf_grad(::Mask, value::Set, args...) = (nothing,)
