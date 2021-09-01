export InertiaModel

################################################################################
# Model definition
################################################################################

"""
Model that uses inertial change points to "explain" interactions
"""
@with_kw struct InertiaModel <: AbstractDynamicsModel

    # stochastic motion parameters
    vel::Float64 = 10 # base vel
    bern::Float64 = 0.9
    k_min::Float64 = 0.5 # min von_misses kappa for angle
    k_max::Float64 = 100.0 # max von_misses kappa for angle
    w_min::Float64 = 2.5 # min standard deviation for magnitude noise
    w_max::Float64 = 5.5 # max standard deviation for magnitude noise

    # force parameters
    # wall rep-> *
    wall_rep_m::Float64 = 0.0
    wall_rep_a::Float64 = 0.02
    wall_rep_x0::Float64 = 0.0
end

function load(::Type{InertiaModel}, path::String)
    InertiaModel(;read_json(path)...)
end

function dynamics_init(dm::InertiaModel, gm::GMParams,
                       cg::CausalGraph, things)
    cg = deepcopy(cg)

    ws = init_walls(gm.area_width, gm.area_height)
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))

    for thing in things
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        set_prop!(cg, v, :object, thing)
    end

    return cg
end

function dynamics_update(dm::InertiaModel,
                         prev_cg::CausalGraph,
                         things)
    cg = get_init_cg(prev_cg)

    gm = get_gm(cg)
    ws = init_walls(gm.area_width, gm.area_height)
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_prop!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))

    for thing in things
        add_vertex!(cg)
        v = MetaGraphs.nv(cg)
        set_prop!(cg, v, :object, thing)
    end
    dynamics_update!(cg, dm)
    graphics_update(cg, prev_cg)
    return cg
end

include("helpers.jl")
include("dynamics.jl")
include("gen.jl")
