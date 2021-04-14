export AbstractGMParams

abstract type AbstractGMParams end

include("utils/params.jl")
include("dynamics_models/dynamics_models.jl")
include("graphics/graphics.jl")
include("utils/flow_masks.jl")
include("utils/state.jl")
include("utils/get_masks_params.jl")

include("inertia/gm_inertia.jl")
include("isr/gm_isr.jl")
include("squishy/gm_squishy.jl")

include("utils/receptive_fields.jl")
include("receptive_fields/gm_receptive_fields.jl")

include("data_generating_procedures/data_generating_procedures.jl")

