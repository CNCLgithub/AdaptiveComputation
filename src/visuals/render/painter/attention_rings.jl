export AttentionRingsPainter

@with_kw struct AttentionRingsPainter <: Painter
    radius::Float64 = 40.0
    attention_color = "red"
    opacity::Float64 = 1.0
    max_attention::Float64 = 4.0
    linewidth::Float64 = 40.0
end

function paint(p::AttentionRingsPainter, d::Dot, attention_weight::Float64)
    norm_weight = attention_weight ./ p.max_attention
    width = 50.0
    ring_weight = norm_weight * width
    pattern = [ring_weight, width - ring_weight]
    # @show pattern
    _draw_circle(get_pos(d), p.radius, p.attention_color;
                 opacity=0.01,
                 style=:stroke, pattern="solid",
                 line=p.linewidth)
    _draw_circle(get_pos(d), p.radius, p.attention_color;
                 opacity=p.opacity*norm_weight,
                 style=:stroke, pattern=pattern,
                 line=p.linewidth)
end
