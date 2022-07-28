export KinPainter

@with_kw struct KinPainter <: Painter
    vel_scale::Float64 = 5.0
    vel_color::String = "black"
    linewidth::Float64 = 7.5
    alpha::Float64 = 1.0
end

function paint(p::KinPainter, d::Dot)
    mag = p.vel_scale * d.vel
    _draw_arrow(d.pos, d.pos .+ mag .+ 1e-3, p.vel_color,
                linewidth = p.linewidth, opacity = p.alpha)
    return nothing
end
