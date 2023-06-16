export KinPainter

@with_kw struct KinPainter <: Painter
    color::Vector
    vel_scale::Float64 = 5.0
    linewidth::Float64 = 7.5
    alpha::Float64 = 1.0
    tail::Bool = false
end

function paint(p::KinPainter, d::Dot, v::Int64)
    mag = p.vel_scale * d.vel
    pos = get_pos(d)
    _draw_arrow(pos, pos .+ mag .+ 1e-3, p.color[v],
                linewidth = p.linewidth, opacity = p.alpha)
    if p.tail
        t = length(d.gstate)
        for k = 1:t
            alpha = p.alpha * exp(-0.5 * (k - 1))
            c::GaussianComponent{2} = d.gstate[k]
            _draw_circle(c.mu, 0.2 * c.cov[1,1], p.color[v],
                         opacity = alpha)
        end
    end
    return nothing
end
