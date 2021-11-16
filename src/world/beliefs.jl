export Diff

import Base.merge


# Vertices, Edges, Graph
const ChangeSrc = Union{Int64, Edge, Symbol}
const StaticPath = Pair{ChangeSrc, Symbol}
const ChangeDiff = Pair{ChangeSrc, Symbol}
const ChangeDict = Dict{ChangeDiff, Any}

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
        [a.born; b.born],
        union(a.died, b.died),
        union(a.static, safe_static),
        merge(a.changed, b.changed)
    )
end


const Registry = Dict{ChangeSrc, ChangeSrc}

function search_registry!(r::Registry, cg::CausalGraph, v::Int64)
    dst = get(r, v, 0)
    dst !== 0 && return dst
    add_vertex!(cg)
    dst = MetaGraphs.nv(cg)
    r[v] = dst
    return dst
end
function search_registry!(r::Registry, cg::CausalGraph, e::Edge)
    dst = getkey(r, e, 0)
    dst !== 0 && return dst
    de = Edge(search_registry!(cg, r, src(e)),
              search_registry!(cg, r, dst(e)))
    add_edge!(cg, de)
    r[e] = de
    return de
end
function search_registry!(r::Registry, cg::CausalGraph, src::Symbol)
    return src
end

"""
apply diff to graph resulting in new graph
"""
function patch(prev_cg::CausalGraph, diff::Diff)

    @unpack static, died, changed, born = diff

    # initialize causal graph
    cg = zero(prev_cg)

    # src -> dst
    registry = Registry()
    # all survived things are copied
    # these are typically walls
    for (src, prop) in static
        in(src, died) && continue
        dst = search_registry!(registry, cg, src)
        # println("$src : $prop -> $dst")
        @>> read(prev_cg, src, prop) patch!(cg, dst, prop)
    end

    # things that have changed but existed before
    for ((src, prop), val) in changed
        in(src, died) && continue
        dst = search_registry!(registry, cg, src)
        # println("$src => $prop : $val -> $dst")
        patch!(cg, dst, prop, val)
    end

    # new things are added
    for ps in born
        add_vertex!(cg)
        dst = MetaGraphs.nv(cg)
        # println("$ps -> $dst")
        set_prop!(cg, dst, :object, ps)
    end

    return cg
end

function read(g::CausalGraph, src::Symbol, prop::Symbol)
    get_prop(g, src)
end
function read(g::CausalGraph, src::Union{Int64, Edge}, prop::Symbol)
    get_prop(g, src, prop)
end

function patch!(g::CausalGraph, dst::Symbol, prop::Symbol, val)
    set_prop!(g, dst, val)
end
function patch!(g::CausalGraph, dst::Union{Int64, Edge}, prop::Symbol, val)
    set_prop!(g, dst, prop, val)
end

"""
apply diff to graph in place

Only `changes` considered.
Can change values of exisitng vertex properties or adding edges
Adding and removing vertices not supported.
"""
function patch!(cg::CausalGraph, diff::Diff)
    @unpack changed = diff
    registry = Registry()
    for v in LightGraphs.vertices(cg)
        registry[v] = v
    end
    for e in LightGraphs.edges(cg)
        registry[e] = e
    end
    # things that have changed but existed before
    for ((s, prop), val) in changed
        isa(s, Edge) && !has_edge(cg, s) &&
            add_edge!(cg, src(s), dst(s))
        patch!(cg, s, prop, val)
    end
    return nothing
end
