export KinPainter


@with_kw struct KinPainter <: Painter
    vel_scale::Float64 = 10.0
    vel_color::String = "black"
    linewidth::Float64 = 7.5
end

function paint(p::KinPainter, cg::CausalGraph, v::Int64, d::Dot)
    mag = p.vel_scale * d.vel
    _draw_arrow(d.pos[1:2], d.pos[1:2] .+ mag .+ 1e-3, p.vel_color,
                linewidth = p.linewidth)
    return nothing
end
