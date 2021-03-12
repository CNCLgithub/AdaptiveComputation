export SubsetPainter

@with_kw struct SubsetPainter <: Painter
    reduction::Function
    painter::Painter
end

function paint(p::SubsetPainter, cg::CausalGraph)
    sg, _ = induced_subgraph(cg, p.reduction(cg))
    paint(p.painter, sg)
    return nothing
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
