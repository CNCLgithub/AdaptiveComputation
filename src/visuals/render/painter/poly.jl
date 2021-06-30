export PolyPainter


@with_kw struct PolyPainter <: Painter
    centroid_color::String = "blue"
    edge_color::String = "black"
    radius::Float64 = 20.0
    show_centroid::Bool = false
end


function paint(p::PolyPainter, cg::CausalGraph, v::Int64, po::Polygon)

    vs = vertices(cg, v)
    #length(vs) == 1 && return # we don't want no UGons

    positions = @>> vs begin
        map(v -> get_prop(cg, v, :object))
        map(o -> get_pos(o))
    end

    # connecting vertices (last one connects 1 -> end)
    inds = @>> 1:length(positions)-1 begin
        map(i -> (i, i+1))
    end
    push!(inds, (1, length(positions)))

    p.show_centroid && _draw_circle(po.pos[1:2], p.radius/2, p.centroid_color)
    @>> inds begin
        foreach(ind -> _draw_arrow(positions[ind[1]][1:2] + rand(2).*1e-4, positions[ind[2]][1:2],
                                   p.edge_color, opacity=1.0, linewidth=2.0,
                                   arrowheadlength=0.0))
    end
    return nothing
end
