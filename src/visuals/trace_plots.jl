export plot_score,
    heatmap,
    plot_rejuvenation,
    plot_attention,
    plot_xy,
    plot_compute_weights

using Gadfly
Gadfly.push_theme(Theme(background_color = colorant"white"))
using Compose
import Cairo
using DataFrames

"""
Plots the log scores in a 2D histogram.
"""
function plot_score(log_weights)
    println("plotting log scores...")
    timesteps = size(log_weights, 1)
    num_particles = size(log_weights, 2)

    x = Array{Int}(undef, timesteps, num_particles)
    for t=1:timesteps
        x[t,:] .= t
    end
    p = plot(x=x, y=log_weights, Geom.histogram2d, Theme(background_color="white"))
    Gadfly.draw(SVG("scores.svg", 6Gadfly.inch, 4Gadfly.inch), p)
end


"""
Creates a heatmap based on z values for given (x,y).
"""
function heatmap(df, x, y, z; points=false)
    println("creating heatmap...")
    p = plot(df, x=x, y=y, color=z,
             points ? Geom.point : Geom.rectbin,
             Scale.color_continuous(minvalue=0.5),
             Theme(background_color="white"))
    Gadfly.draw(SVG("heatmap.svg", 6Gadfly.inch, 4Gadfly.inch), p)
end

function plot_compute_weights(weights::Matrix{Float64}, path::String;
                              tracker_colors = TRACKER_COLORSCHEME)
    # weights = clamp.(weights, -1E6, 0)
    k,n = size(weights)
    ts = repeat(1:k, 1, size(weights, 2))
    data = zeros(k)
    @inbounds for t = 1:k
        data[t] = logsumexp(weights[t, :]) - log(n)
    end
    plt = plot(
        x = 1:k,
        y = data,
        Geom.line,
        Theme(background_color = "white"))
    out = joinpath(path, "compute_weights.png")
    plt |> PNG(out, √200Gadfly.cm, 20Gadfly.cm; dpi=96)
end

"""
Plots rejuvenation steps accross time
"""
function plot_rejuvenation(rejuvenations, max_sweeps, path="plots")
    mkpath(path)
    k = length(rejuvenations)
    x = collect(1:k)
    
    p = plot(x=x, y=rejuvenations,
             Geom.bar,
             Scale.x_continuous(minvalue=0, maxvalue=k),
             Scale.y_continuous(minvalue=0, maxvalue=max_sweeps),
             Guide.xlabel("Time"),
             Guide.ylabel("Allocated Compute"),
             Theme(default_color="black",
                   background_color="white",
                   minor_label_font_size=20pt,
                   major_label_font_size=30pt)
             )
    Gadfly.draw(PNG(joinpath(path, "rejuvenations.png"), 16Gadfly.inch, 6Gadfly.inch), p)
end

"""
Plots distribution of attention accross time
"""
function plot_attention(attended, max_sweeps::Int, path::String;
                        tracker_colors=TRACKER_COLORSCHEME)
    mkpath(path)

    n_trackers, k = size(attended)
    if n_trackers > length(tracker_colors)
        tracker_colors = ColorScheme(distinguishable_colors(n_trackers))
    end
    x = collect(1:k)
    plots = []
    for i = 1:n_trackers

        att_tracker = attended[i, :]

        p = plot(x=x, y=att_tracker,
                 Geom.bar,
                 Scale.y_continuous(minvalue=0, maxvalue=max_sweeps),
                 Theme(default_color=tracker_colors[i],
                       background_color="white")
                 )

        push!(plots, p)
    end

    p = vstack(Tuple(plots)...)

    Gadfly.draw(PNG(joinpath(path, "attention.png"), 4Gadfly.inch, n_trackers*2Gadfly.inch), p)
end

"""
Plots the particle filter position estimates over time
"""
function plot_xy(xy)
    folder = "position_estimates"
    mkpath(folder)

    T = size(xy, 1)
    num_samples = size(xy, 2)

    for t=1:T
        p = plot(x=xy[t,:,:,1],
                 Geom.histogram(bincount=400),
                 Scale.x_continuous(minvalue=-200.0, maxvalue=200.0),
                 Scale.y_continuous(minvalue=0.0, maxvalue=num_samples/10),
                 Theme(background_color="white")
                 )
        Gadfly.draw(PNG("$folder/$(lpad(t, 3, "0")).png", 8Gadfly.inch, 3Gadfly.inch), p)
    end
end
