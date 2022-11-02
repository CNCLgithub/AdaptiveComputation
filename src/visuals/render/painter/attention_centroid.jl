export AttentionCentroidPainter


@with_kw struct AttentionCentroidPainter <: Painter
    tau::Float64 = 1.0
    color::String = "red"
    radius::Float64 = 30.0
    opacity::Float64 = 0.8
end

function paint(p::AttentionCentroidPainter, st::InertiaState, attention_weights::Vector{Float64})
    points = @>> get_objects(st) begin
        map(get_pos)
        x -> (hcat(x...)')
        x -> Matrix{Float64}(x)
    end
    n = length(attention_weights)
    aws = sum(attention_weights)
    if aws <= 0.
        aw = fill(1.0 / n, n)
    else
        aw = attention_weights ./ sum(attention_weights)
    end
    weighted_mean = vec(sum(points .* aw, dims=1))
    _draw_circle(weighted_mean, p.radius, p.color; opacity=p.opacity)
end
