abstract type Flow{T<:Space} end
abstract type NullFlow end

evolve(::Flow, ::Space) = error("not implemented")

# with respect to one object
@with_kw struct ExponentialFlow{T} <: Flow{T}
    decay_rate::Float64
    exp_dr::Float64 = exp(decay_rate)
    memory::T
    lower::Float64 = 1E-5
end

function ExponentialFlow(flow::ExponentialFlow{T}, space::T) where {T <: Space}
    # decay memory
    decayed = flow.memory * flow.exp_dr

    # @inbounds for i in eachindex(decayed)
    #     decayed[i] = round(decayed[i]; digits = 5)
    # end
    # decayed = round.(decayed, digits = 5)
    # map!(_round, decayed, decayed)
    # dropzeros!(decayed)
    droptol!(decayed, flow.lower)

    memory = max.(space, decayed)

    ExponentialFlow{T}(flow.decay_rate, flow.exp_dr,
                       memory, flow.lower)
end

evolve(flow::ExponentialFlow{T}, space::T) where {T <: Space} = ExponentialFlow(flow, space)


function _round(cell::Float64; digits::Int64 = 5)
    round(cell, digits = digits)
end
# render(flow::ExponentialFlow) = clamp.(flow.memory, 0.0, 1.0)
