using ColorSchemes
using ColorTypes
using Colors
const TRACKER_COLORSCHEME = colorschemes[:Set1_9]

include("render/renderv2.jl")
include("trace_plots.jl")
include("render_scene.jl")
include("render_masks.jl")
#TODO: depricate
# include("render_rf_masks.jl")
include("visualize_inference.jl")
