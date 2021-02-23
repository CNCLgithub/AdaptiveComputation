export AbstractCausalGraph, CausalGraph
using LightGraphs
using MetaGraphs


abstract type AbstractCausalGraph end


struct CausalGraph{T, G<:AbstractGraph} <: AbstractCausalGraph
    elements::AbstractArray{T}
    graph::G
end

function CausalGraph(elements::AbstractArray{T}, g::Type{G}) where {T, G <:AbstractGraph}
    graph = g(length(elements))
    CausalGraph{T, g}(elements, graph)
end


function update(cg::CausalGraph{T, G}, els::AbstractArray{T}) where {T,G}
    CausalGraph{T, G}(els, cg.graph)
end
