export AttentionGaussianPainter


@with_kw struct AttentionGaussianPainter <: Painter
    area_dims::Tuple{Int64, Int64} = (500, 500)
    dims::Tuple{Int64, Int64} = (500, 500)
    attention_color::String = "red"
    opacity::Float64 = 0.7
end


function paint(p::AttentionGaussianPainter, cg::CausalGraph, attention_weights::Vector{Float64})
    points = @>> get_objects(cg, Dot) begin
        map(x -> x.pos[1:2])
        x -> (hcat(x...)')
        x -> Matrix{Float64}(x)
    end

    #attention_weights = fill(1.0, size(points,1))
    @show points
    norm_weights = exp.(attention_weights)/sum(exp.(attention_weights))

    @show attention_weights
    @show norm_weights

    #weighted_mean = vec(mean(points .* norm_weights, dims=1))
    weighted_mean = vec(sum(points .* norm_weights, dims=1))
    #weighted_mean[1] = -weighted_mean[1]

    weighted_cov = cov(points .* norm_weights, dims=1)
    #weighted_cov = [100 0; 0 100]
    
    @show weighted_mean
    @show weighted_cov
    
    values = Matrix{Float64}(undef, p.dims[1], p.dims[2])

    for i=1:p.dims[1], j=1:p.dims[2]
        x = i - p.dims[1]/2
        y = j - p.dims[2]/2
        values[i,j] = exp(Gen.logpdf(mvnormal, [x, y], weighted_mean, weighted_cov))
    end
    
    values /= maximum(values)
    values = reverse(values, dims=2)'
    @show maximum(values)

    _draw_array(values, p.area_dims... , p.dims..., p.attention_color, opacity=p.opacity)
end
