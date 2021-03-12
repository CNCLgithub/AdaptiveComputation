export IFPainter

@with_kw struct IFPainter <: Painter
    force_scale::Float64 = 10.0
    force_color::String = "black"
end


function paint(p::IFPainter, cg::CausalGraph, e::Edge)

    vs = [src(e), dst(e)]
    positions = @>> vs begin
        map(v -> get_prop(cg, v, :object))
        map(o -> get_pos(o))
    end
    force = get_prop(cg, e, :force)
    force_mag = norm(force) / p.force_scale

    positions[2] .+= fill(1e-3, 3) # so that positions[1] != positions[2]

    _draw_arrow(positions[1][1:2], positions[2][1:2],
                p.force_color, opacity=1.0,
                linewidth=force_mag,
                arrowheadlength=0.0)
    return nothing
end
