export AttentionRingsPainter

@with_kw struct AttentionRingsPainter <: Painter
    radius::Float64 = 40.0
    attention_color = "red"
    opacity::Float64 = 1.0
    max_attention::Float64 = 4.0
    linewidth::Float64 = 7.0
end

function paint(p::AttentionRingsPainter, d::Dot, attention_weight::Float64)
    norm_weight = attention_weight ./ p.max_attention
    _draw_circle(d.pos, p.radius, p.attention_color;
                 opacity=p.opacity*norm_weight,
                 style=:stroke, pattern="longdashed", line=p.linewidth)
end
