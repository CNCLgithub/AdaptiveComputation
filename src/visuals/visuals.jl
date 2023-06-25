export render_trace, render_pf

using ImageIO
using ColorTypes
using Colors
import ColorBlendModes; # instead of `using`
using ColorBlendModes.BlendModes, ColorBlendModes.CompositeOperations;

# const TRACKER_COLORSCHEME = colorschemes[:Set1_9]


function render_trace(tr::Gen.Trace, path::String)
    (t, gm) = get_args(tr)
    render_trace(gm, tr, path)
    return nothing
end

function render_pf(chain::PFChain, path::String)
    @unpack state, auxillary = chain
    (t, gm) = get_args(first(state.traces))
    render_pf(gm, chain, path)
end


include("render/render.jl")
include("gm_inertia.jl")
include("visualize_inference.jl")

# include("trace_plots.jl")
# include("render_scene.jl")
# #TODO: depricate
# # include("render_rf_masks.jl")
