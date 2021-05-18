export FixationsPainter


@with_kw struct FixationsPainter <: Painter
    fixations_color::String = "black"
    fixations_radius::Float64 = 5.0
    fixations_opacity::Float64 = 0.7
end

function paint(p::FixationsPainter, fixations::Array{Float64, 2})
    # going through subjects
    for i=1:size(fixations, 1)
        _draw_circle(fixations[i,:], p.fixations_radius,
                     p.fixations_color, opacity=p.fixations_opacity)
    end
end

function paint(p::FixationsPainter, fixations::Array{Float64, 3})
    # going through subjects
    nt = size(fixations, 1)
    for i=1:size(fixations, 2)
        for k = nt:-2:2
            start = fixations[k,i,:]
            stop = fixations[k - 1,i,:]
            _draw_line(start, stop,
                       p.fixations_color, opacity=p.fixations_opacity)
        end
    end
end
