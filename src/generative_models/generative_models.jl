export AbstractGMParams, GMParams,
    AbstractDynamicsModel, dynamics_init, dynamics_update

################################################################################
# Interface
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
# Causation
################################################################################

function causal_init(gm::AbstractGMParams, dm::AbstractDynamicsModel,
                     gr::AbstractGraphics, things::AbstractArray{Thing})
    cg = get_init_cg(gm, dm, gr)
    dynamics_init!(cg, things)
    # no graphics since initial state is hidden
    return cg
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

function causal_update(prev_cg::CausalGraph, diff)::CausalGraph

    # initialize causal graph
    cg = CausalGraph(SimpleDiGraph())

    # copy meta data from previous graph
    @>> prev_cg props set_props!(cg)

    # apply diff to graph
    @unpack static, changed, born = diff

    # all survived things are copied
    # these are typically walls
    # (dead objects are removed implicitly)
    for src in static
        add_vertex!(cg)
        dst = MetaGraphs.nv(cg)
        set_prop!(cg, dst, :object,
                  get_prop(prev_cg, src, :object))
    end

    # things that have changed but existed before
    for (src, thing) in changed
        add_vertex!(cg)
        dst = MetaGraphs.nv(cg)
        set_prop!(cg, dst, :object, thing)
    end

    # new things are added
    for thing in born
        add_vertex!(cg)
        dst = MetaGraphs.nv(cg)
        set_prop!(cg, dst, :object, thing)
    end

    # resolve lower causal processes
    dynamics_update!(cg)
    graphics_update!(cg)

    return cg
end

################################################################################
# Dynamics
################################################################################

abstract type AbstractDynamicsModel end

function dynamics_init!(cg::CausalGraph, trackers::AbstractArray{Thing})
    dynamics_init!(cg, get_dm(cg), get_gm(cg), trackers)
end
function dynamics_init!(::CausalGraph, ::AbstractDynamicsModel,
                        ::AbstractArray{Thing})
    error("not implemented")
end

function dynamics_update!(cg::CausalGraph)
    dynamics_update(get_dm(cg), cg)
end
function dynamics_update!(::AbstractDynamicsModel, cg::CausalGraph)
    error("not implemented")
end
################################################################################
# Graphics
################################################################################

abstract type AbstractGraphics end

load(::Type{AbstractGraphics}) = error("not implemented")
get_observations(::AbstractGraphics) = error("not implemented")

function graphics_init!(cg::CausalGraph)::CausalGraph
    graphics_init!(cg, get_graphics(cg))
end

function graphics_init!(::CausalGraph, ::AbstractGraphics)::CausalGraph
    error("not implemented")
end

function graphics_update!(cg::CausalGraph, prev_cg::CausalGraph)::CausalGraph
    graphics_update(cg, get_graphics(cg), prev_cg)
end

function graphics_update(::CausalGraph, ::AbstractGraphics,
                         ::CausalGraph)::CausalGraph
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

