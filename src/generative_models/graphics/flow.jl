abstract type Flow{T<:Space} end
abstract type NullFlow end

evolve(::Flow, ::Space) = error("not implemented")

# with respect to one object
@with_kw struct ExponentialFlow{T} <: Flow{T}
    decay_rate::Float64
    memory::T
    upper::Float64 = 0.8
end

function ExponentialFlow(flow::ExponentialFlow{T}, space::T) where {T <: Space}
    memory = flow.memory * exp(flow.decay_rate)
    memory += space
    memory *= 0.5
    clamp!(memory, 0.0, flow.upper)
    ExponentialFlow{T}(flow.decay_rate, memory, flow.upper)
end

evolve(flow::ExponentialFlow{T}, space::T) where {T <: Space} = ExponentialFlow(flow, space)

render(flow::ExponentialFlow) = clamp.(flow.memory, 0.0, 1.0)
