export AbstractGMParams, GMParams,
    AbstractDynamicsModel, dynamics_update

################################################################################
# Generative Model specifications
################################################################################

abstract type AbstractGMParams end


function tracker_bounds(cg::CausalGraph)
    tracker_bounds(get_gm(cg))
end

function tracker_bounds(::AbstractGMParams)
    error("Not implemented")
end

include("gm_params.jl")


################################################################################
# Dynamics
################################################################################

abstract type AbstractDynamicsModel end

"""
Defines how a dynamics model changes dynamic properties (such as forces).
"""
function dynamics_update(::AbstractDynamicsModel, ::CausalGraph)::Diff
    error("not implemented")
end
################################################################################
# Graphics
################################################################################

abstract type AbstractGraphics end

load(::Type{AbstractGraphics}) = error("not implemented")
get_observations(::AbstractGraphics) = error("not implemented")

function render(::AbstractGraphics,
                ::CausalGraph)::Diff
    error("not implemented")
end
function predict(::AbstractGraphics,
                 ::CausalGraph)::Diff
    error("not implemented")
end

include("graphics/graphics.jl")


################################################################################
# Causation
################################################################################

function causal_init(::AbstractGMParams, ::AbstractDynamicsModel,
                     ::AbstractGraphics, ::AbstractArray{Thing})
    error("not implemented")
end

"""
Every model defines a ladder of causation that
in turn defines how beliefs over the world change
across time steps
"""
function causal_update(::AbstractDynamicsModel, ::CausalGraph,
                       ::Diff)::CausalGraph
    error("not implemented")
end


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

