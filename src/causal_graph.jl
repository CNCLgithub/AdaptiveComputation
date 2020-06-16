export AbstractCausalGraph, CausalGraph

using LightGraphs

abstract type AbstractCausalGraph end

struct CausalGraph{T, AbstractGraph} <: AbstractCausalGraph
    elements::Vector{T}
    graph::AbstractGraph
    function CausalGraph(elements::Vector{T}, g::Type{AbstractGraph}) where {T}
        graph = g(length(elements))
        new{T, g}(elements, graph)
    end
end
