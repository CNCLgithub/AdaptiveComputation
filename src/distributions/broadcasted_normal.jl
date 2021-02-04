"""
    my own broadcasted_normal because GenRFS is complaining:
    "Cannot `convert` an object of type Gen.BroadcastedNormal to an object of type Gen.Distribution{Array}"
"""

export my_broadcasted_normal

struct MyBroadcastedNormal <: Gen.Distribution{Array} end

const my_broadcasted_normal = MyBroadcastedNormal()

function Gen.random(::MyBroadcastedNormal, mus::Vector{Float64}, sigmas::Vector{Float64})
    @>> 1:length(mus) map(i -> normal(mus[i], sigmas[i]))
end

function Gen.logpdf(::MyBroadcastedNormal, x::Vector{Float64},
                    mus::Vector{Float64}, sigmas::Vector{Float64})
    lpdf = @>> 1:length(x) map(i -> Gen.logpdf(normal, x[i], mus[i], sigmas[i])) sum
end

(::MyBroadcastedNormal)(mus, sigmas) = Gen.random(MyBroadcastedUniform(), mus, sigmas)

Gen.has_output_grad(::MyBroadcastedNormal) = false
Gen.logpdf_grad(::MyBroadcastedNormal, value::Set, args...) = (nothing,)
