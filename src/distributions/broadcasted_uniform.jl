export broadcasted_uniform

struct BroadcastedUniform <: Gen.Distribution{Array} end

const broadcasted_uniform = BroadcastedUniform()

function Gen.random(::BroadcastedUniform, lows::Vector{Float64}, highs::Vector{Float64})
    @>> 1:length(lows) map(i -> uniform(lows[i], highs[i]))
end

function Gen.logpdf(::BroadcastedUniform, x::Vector{Float64},
                    lows::Vector{Float64}, highs::Vector{Float64})
    @>> 1:length(x) map(i -> Gen.logpdf(uniform, x[i], lows[i], highs[i])) sum
end

(::BroadcastedUniform)(lows, highs) = Gen.random(BroadcastedUniform(), lows, highs)

Gen.has_output_grad(::BroadcastedUniform) = false
Gen.logpdf_grad(::BroadcastedUniform, value::Set, args...) = (nothing,)
