export PoissDotPainter


@with_kw struct PoissDotPainter <: Painter
    rate::Float64 = 2.0
    dot_color::String = "#b4b4b4"
end


function paint(p::PoissDotPainter, cg::CausalGraph, v::Int64, d::Dot)
    rvargs = (fill(0., 2), fill(d.radius, 2))
    ppp = RFSElements{Array{Float64}}(undef, 1)
    ppp[1] = PoissonElement{Array{Float64}}(p.rate, broadcasted_normal, rvargs)
    positions = rfs(ppp)
    for pos in positions
        _draw_circle(d.pos[1:2] + pos, d.radius, p.dot_color)
    end
    return nothing
end
