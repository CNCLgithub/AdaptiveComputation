abstract type Flow{T<:Space} end

evolve(::Flow, ::Space) = error("not implemented")

# with respect to one object
@with_kw struct ExponentialFlow{T} <: Flow{T}
    decay_rate::Float64
    memory::T
end

function ExponentialFlow(flow::ExponentialFlow{T}, space::T) where {T <: Space}
    memory = flow.memory * exp(flow.decay_rate)
    memory += space
    ExponentialFlow{T}(flow.decay_rate, memory)
end

evolve(flow::ExponentialFlow{T}, space::T) where {T <: Space} = ExponentialFlow(flow, space)

render(flow::ExponentialFlow) = clamp.(flow.memory, 0.0, 1.0)
