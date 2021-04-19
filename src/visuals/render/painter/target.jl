export TargetPainter

@with_kw struct TargetPainter <: Painter
    target_color::String = "#ea3455"
    targets::Vector{Bool}
end

function paint(p::TargetPainter, cg::CausalGraph)
    dots = get_objects(cg, Dot)
    dots = dots[p.targets]
    for (i, d) in enumerate(dots)
        paint(p, cg, i, d)
    end
    return nothing
end

function paint(p::TargetPainter, cg::CausalGraph, v::Int64, d::Dot)
    _draw_circle(d.pos[1:2], d.radius, p.target_color)
end
