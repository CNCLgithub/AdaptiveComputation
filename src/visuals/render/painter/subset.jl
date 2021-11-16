export SubsetPainter

@with_kw struct SubsetPainter <: Painter
    reduction::Function
    painter::Painter
end

remap(p::Painter, vmap::Vector{Int64})::Painter = p

function paint(p::SubsetPainter, cg::CausalGraph)
    sg, vmap = induced_subgraph(cg, p.reduction(cg))
    @> p.painter remap(vmap) paint(sg)
    return nothing
end

function only_targets(cg::CausalGraph)
    @>> Dot begin
        get_object_verts(cg)
        filter(v -> target(get_prop(cg, v, :object)) > 0)
    end
end
function only_targets(cg::CausalGraph, ts::BitArray{1})
    vs = get_object_verts(cg, Dot)
    vs = collect(vs)
    vs[ts]
end

function only_targets(cg::CausalGraph, ts::Vector{Bool})
    vs = get_object_verts(cg, Dot)
    vs = collect(vs)
    vs[ts]
end
