export GenerativeModel, GMState

abstract type GenerativeModel end
abstract type GMState{T<:GenerativeModel} end


################################################################################
# Generative Model specifications
################################################################################
"""
    step(::GenerativeModel, ::GMState)

Evolves state according rules in generative model
"""
function step end

"""
    observe(::GenerativeModel, ::GMState)

Observe given the latent state returning a tuple
1. Elements parameterizing the random finite set
2. Observed set
"""
function observe end

"""
    predict(::GenerativeModel, ::GMState)

Return the elements parameterizing the random finite set
"""
function predict end

################################################################################
# Generative models
################################################################################

include("things.jl")
include("inertia/inertia.jl")
include("isr/isr.jl")
include("force/force.jl")
include("force_ensemble/force_ensemble.jl")

################################################################################
# Data generating procedures
################################################################################

# include("data_generating_procedures/data_generating_procedures.jl")
