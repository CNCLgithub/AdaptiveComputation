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
    nt, nsub, _ = size(fixations)

    for i=reverse(1:nt)
        for j=1:nsub
            a = max(i-1, 1)
            start = fixations[a,j,:]
            stop = fixations[i,j,:]
            _draw_line(start, stop, p.fixations_color,
                       opacity=p.fixations_opacity)
        end
    end
end
