export AbstractDynamicsModel

abstract type AbstractDynamicsModel end


function dynamics_init(cg::CausalGraph, trackers::Vector{Thing})
    dynamics_init(get_dm(cg), get_gm(cg), cg, trackers)
end

function dynamics_update(cg::CausalGraph, trackers::Vector{Thing})
    dynamics_update(get_dm(cg), cg, trackers)
end


#include("brownian_dynamics_model.jl")
#include("cbm.jl")
#include("isr_pylons.jl")
#include("radial_motion.jl")
#include("hgm_dynamics_model_v2.jl")
include("squishy/squishy.jl")
include("isr/isr.jl")
include("inertia/inertia.jl")
