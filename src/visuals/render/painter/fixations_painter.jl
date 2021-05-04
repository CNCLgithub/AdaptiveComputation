export FixationsPainter


@with_kw struct FixationsPainter <: Painter
    fixations_color::String = "black"
    fixations_radius::Float64 = 5.0
    fixations_opacity::Float64 = 0.7
end


function paint(p::FixationsPainter, fixations::Matrix{Float64})
    # going through subjects
    for i=1:size(fixations, 1)
        _draw_circle(fixations[i,:], p.fixations_radius,
                     p.fixations_color, opacity=p.fixations_opacity)
    end
end
