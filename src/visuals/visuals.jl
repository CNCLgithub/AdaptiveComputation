using Plots: palette # for the color scheme
default_colors = palette(:tab20)

include("render.jl")
include("visualize_inference.jl")
include("trace_plots.jl")
