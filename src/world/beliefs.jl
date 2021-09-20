export Diff

import Base.merge

struct Diff
    # persistence
    born::Vector{Thing}
    died::Vector{Int64}
    # mutability
    static::Vector{StaticPath}
    changed::Dict{ChangeDiff, Any}
end

function Diff(cg::CausalGraph, born::Vector{Thing},
              died::Vector{Int64})
    Diff(born, died, Int64[], Dict{Int64, Thing}())
end
function Diff(changed::Dict{ChangeDiff})
    Diff(Thing[], Int64[], StaticPath[], changed)
end
function Diff(st::Vector{StaticPath})
    Diff(Thing[], Int64[], st, Dict{ChangeDiff, Any}())
end

function Base.merge(a::Diff, b::Diff)
    safe_static = filter(x -> !in(first(x), a.died), b.static)
    Diff(
        [a.born, b.born],
        union(a.died, b.died),
        union(a.static, safe_static),
        merge(a.changed, b.changed)
    )
end


# Vertices, Edges, Graph
const ChangeSrc = Union{Int64, Edge, Symbol}
const ChangeDiff = Pair{ChangeSrc, Symbol}
const StaticPath = Pair{ChangeSrc, Symbol}
