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

const Registry = Dict{ChangeSrc, ChangeSrc}

function search_registry!(r::Registry, cg::CausalGraph, src::Int64)
    dst = getkey(r, src, 0)
    dst !== 0 && return dst
    add_vertex!(cg)
    dst = MetaGraphs.nv(cg)
    r[v] = dst
    return dst
end
function search_registry!(r::Registry, cg::CausalGraph, e::Edge)
    dst = getkey(r, e, 0)
    dst !== 0 && return dst
    de = Edge(search_registry!(cg, r, src(e)),
              search_registry!(cg, r, dst(e)))
    add_edge!(cg, de)
    r[e] = de
    return de
end
function search_registry!(r::Registry, cg::CausalGraph, src::Symbol)
    return src
end

"""
apply diff to graph resulting in new graph
"""
function patch(prev_cg::CausalGraph, diff::Diff)

    @unpack static, died, changed, born = diff

    # initialize causal graph
    cg = zero(prev_cg)

    # src -> dst
    registry = Registry()
    # all survived things are copied
    # these are typically walls
    for (src, prop) in static
        dst = search_registry!(cg, registry, src)
        @>> read(cg, src, prop) patch!(cg, dst, prop, dst)
    end

    # things that have changed but existed before
    for ((src, prop), val) in changed
        dst = search_registry!(cg, registry, src)
        patch!(cg, dst, prop, val)
    end

    # new things are added
    for ps in born
        add_vertex!(cg)
        dst = MetaGraphs.nv(cg)
        set_props!(cg, dst, ps)
    end

    return cg
end

function read(g::CausalGraph, src::Symbol, prop::Symbol)
    get_prop(g, src)
end
function read(g::CausalGraph, src::Union{Int64, Edge}, prop::Symbol)
    get_prop(g, src, prop)
end

function patch!(g::CausalGraph, dst::Symbol, prop::Symbol, val)
    set_prop!(g, dst, val)
end
function patch!(g::CausalGraph, dst::Union{Int64, Edge}, prop::Symbol, val)
    set_prop!(b, dst, prop, val)
end

"""
apply diff to graph in place

Only `changes` considered.
Can change values of exisitng vertex properties or adding edges
Adding and removing vertices not supported.
"""
function patch!(cg::CausalGraph, diff::Diff)
    @unpack changed = diff
    registry = Registry()
    for v in LightGraphs.vertices(cg)
        registry[v] = v
    end
    for e in LightGraphs.edges(cg)
        registry[e] = e
    end
    # things that have changed but existed before
    for ((s, prop), val) in changed
        istype(s, Edge) && !has_edge(cg, s) &&
            add_edge!(cg, src(s), dst(s))
        set_prop!(cg, s, prop, val)
    end
    return nothing
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

function graphics_init!(cg::CausalGraph)::CausalGraph
    graphics_init!(cg, get_graphics(cg))
end

function graphics_init!(::CausalGraph, ::AbstractGraphics)::CausalGraph
    error("not implemented")
end

function graphics_update(cg::CausalGraph)::Diff
    graphics_update(get_graphics(cg), cg)
end

function graphics_update(::AbstractGraphics,
                         ::CausalGraph)::Diff
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

