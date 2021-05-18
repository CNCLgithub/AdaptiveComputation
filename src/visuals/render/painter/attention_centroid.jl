export AttentionCentroidPainter


@with_kw struct AttentionCentroidPainter <: Painter
    tau::Float64 = 1.0
    color::String = "red"
    radius::Float64 = 30.0
    opacity::Float64 = 0.8
end

function paint(p::AttentionCentroidPainter, cg::CausalGraph, attention_weights::Vector{Float64})
    points = @>> get_objects(cg, Dot) begin
        map(x -> x.pos[1:2])
        x -> (hcat(x...)')
        x -> Matrix{Float64}(x)
    end
    
    n_objects = size(points, 1)
    prior_weights = fill(1.0/n_objects, n_objects)

    attention_weights = p.tau * attention_weights + (1.0 - p.tau) * prior_weights .+ 0.01
    norm_weights = attention_weights / sum(attention_weights)
    weighted_mean = vec(sum(points .* norm_weights, dims=1))
    
    _draw_circle(weighted_mean, p.radius, p.color; opacity=p.opacity)
end
