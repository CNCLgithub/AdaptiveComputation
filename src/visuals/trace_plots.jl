export plot_score,
    heatmap,
    plot_rejuvenation,
    plot_xy

using Gadfly
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

"""
Plots rejuvenation steps accross time
"""
function plot_rejuvenation(rejuvenations; t=nothing)
    mkpath("rejuvenations")

    rej_new = deepcopy(rejuvenations)
    file = "rejuvenation_steps"
    if !isnothing(t)
        rej_new[t+1:end] .= 0
        """
        rej_new = rej_new[51:end]
        rej_new = rej_new[1:120]
        if t < 51
            rej_new .= 0
        end
        """

        file = "$(lpad(t, 3, "0"))"
    end

    T = size(rej_new, 1)
    x = collect(1:T)

    p = plot(x=x, y=rej_new,
             Geom.bar,
             Scale.y_continuous(minvalue=0, maxvalue=20),
             Theme(default_color="black",
                   background_color="white")
             )
    Gadfly.draw(SVG("rejuvenations/$file.svg", 8Gadfly.inch, 3Gadfly.inch), p)
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
