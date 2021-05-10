export AttentionRingsPainter

@with_kw struct AttentionRingsPainter <: Painter
    radius::Float64 = 40.0
    attention_color::String = "red"
    opacity::Float64 = 1.0
    max_attention::Float64 = 4.0
    linewidth::Float64 = 7.0
end

function paint(p::AttentionRingsPainter, cg::CausalGraph, attention_weights::Vector{Float64})
    @show attention_weights
    points = @>> get_objects(cg, Dot) map(x -> x.pos[1:2])
    norm_weights = attention_weights/p.max_attention
    
    for (i, point) in enumerate(points)
        _draw_circle(point, p.radius, p.attention_color;
                     opacity=p.opacity*norm_weights[i],
                     style=:stroke, pattern="longdashed", line=p.linewidth)
    end
end
