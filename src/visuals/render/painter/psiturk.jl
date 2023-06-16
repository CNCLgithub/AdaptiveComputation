export PsiturkPainter

@with_kw struct PsiturkPainter <: Painter
    dot_color = "#b4b4b4"
    # highlight = "#ea3433"
    probe_color = "#a0a0a0"
    wall_color = "black"
    alpha::Float64 = 1.0
end

function paint(p::PsiturkPainter, dot::Dot)
    _draw_circle(get_pos(dot), dot.radius, p.dot_color,
                 opacity = p.alpha)
    return nothing
end

function paint(p::PsiturkPainter, w::Wall)
    _draw_arrow(w.p1, w.p2, p.wall_color, arrowheadlength=0.0)
    return nothing
end
