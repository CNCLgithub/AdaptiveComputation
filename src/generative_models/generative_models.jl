abstract type AbstractGMParams end

include("causal_graph.jl")
include("dynamics_models/dynamics_models.jl")
include("graphics/graphics.jl")

include("flow_masks.jl")
include("gm_brownian.jl")
include("gm_inertia.jl")
include("gm_isr.jl")
include("gm_isr_pylons.jl")
include("probe_brownian.jl")
# include("gm_positions_cbm_static.jl")
include("hgm.jl")

include("data_generating_procedures/data_generating_procedures.jl")

