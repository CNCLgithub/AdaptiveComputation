export GenerativeModel, GMState

abstract type GenerativeModel end
abstract type GMState end


################################################################################
# Generative Model specifications
################################################################################
"""
Evolves state according rules in generative model
"""
function step(::GenerativeModel, ::GMState)::GMState
    error("Not implemented")
end

function render(::GenerativeModel, ::GMState)
    error("Not implemented")
end

function predict(::GenerativeModel, ::GMState)
    error("Not implemented")
end

################################################################################
# Generative models
################################################################################

include("things.jl")
include("graphics.jl")
# include("isr/isr.jl")
include("inertia/inertia.jl")

################################################################################
# Data generating procedures
################################################################################

# include("data_generating_procedures/data_generating_procedures.jl")
