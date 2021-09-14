export AbstractGMParams, GMParams,
    AbstractDynamicsModel, dynamics_init, dynamics_update

################################################################################
# Interface
################################################################################

abstract type AbstractGMParams end


function tracker_bounds(cg::CausalGraph)
    tracker_bounds(get_gm(cg), cg)
end

function tracker_bounds(::AbstractGMParams, cg::CausalGraph)
    error("Not implemented")
end

include("gm_params.jl")


################################################################################
# Causation
################################################################################

function init_cg_from_things(cg::CausalGraph, things::AbstractArray{Things})
    @>> things begin
        dynamics_init(cg)
        graphics_init
    end
end

function get_init_cg(cg::CausalGraph)
    get_init_cg(get_gm(cg), get_dm(cg), get_graphics(cg))
end
function get_init_cg(gm::AbstractGMParams, dm::AbstractDynamicsModel)
    get_init_cg(gm, dm, NullGraphics())
end
function get_init_cg(gm::AbstractGMParams, dm::AbstractDynamicsModel,
                     graphics::AbstractGraphics)
    cg = CausalGraph(SimpleDiGraph())
    set_prop!(cg, :gm, gm)
    set_prop!(cg, :dm, dm)
    set_prop!(cg, :graphics, graphics)
    return cg
end

function causal_update(cg::CausalGraph, diff)

end

################################################################################
# Dynamics
################################################################################

abstract type AbstractDynamicsModel end

function dynamics_init(cg::CausalGraph, trackers::AbstractArray{Thing})
    dynamics_init(get_dm(cg), get_gm(cg), cg, trackers)
end
function dynamics_init(::AbstractDynamicsModel, cg::CausalGraph,
                       trackers::AbstractArray{Thing})
    error("not implemented")
end

function dynamics_update(cg::CausalGraph, trackers::AbstractArray{Thing})
    dynamics_update(get_dm(cg), cg, trackers)
end
function dynamics_update(::AbstractDynamicsModel, cg::CausalGraph,
                         trackers::AbstractArray{Thing})
    error("not implemented")
end
################################################################################
# Graphics
################################################################################

abstract type AbstractGraphics end

load(::Type{AbstractGraphics}) = error("not implemented")
get_observations(::AbstractGraphics) = error("not implemented")

function graphics_init(cg::CausalGraph)::CausalGraph
    graphics_init(get_graphics(cg), cg)
end

function graphics_init(::AbstractGraphics, cg::CausalGraph)::CausalGraph
    error("not implemented")
end

function graphics_update(cg::CausalGraph, prev_cg::CausalGraph)::CausalGraph
    graphics_update(get_graphics(cg), cg, prev_cg)
end

function graphics_update(::AbstractGraphics, cg::CausalGraph,
                         prev_cg::CausalGraph)::CausalGraph
    error("not implemented")
end

include("graphics/graphics.jl")

################################################################################
# Inference models
################################################################################

# include("common/common.jl")
include("isr/isr.jl")
include("inertia/inertia.jl")
#include("squishy/gm_squishy.jl")

################################################################################
# Data generating procedures
################################################################################

include("data_generating_procedures/data_generating_procedures.jl")

