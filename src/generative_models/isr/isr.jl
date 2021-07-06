export ISRDynamics

@with_kw struct ISRDynamics <: AbstractDynamicsModel
    repulsion::Bool = true
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance::Float64 = 60.0
    vel::Float64 = 10.0 # base velocity
    rep_inertia::Float64 = 0.9

    brownian::Bool = true
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_x::Float64 = 1.0
    sigma_y::Float64 = 1.0
end

function load(::Type{ISRDynamics}, path::String)
    ISRDynamics(;read_json(path)...)
end

function dynamics_init(dm::ISRDynamics, gm::GMParams,
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

    #cg = dynamics_update!(dm, cg, things)
    return cg
end

function dynamics_update(dm::ISRDynamics,
                         cg::CausalGraph,
                         things)
    cg = deepcopy(cg)
    vs = get_object_verts(cg, Dot)
    for (i, thing) in enumerate(things)
        set_prop!(cg, vs[i], :object, thing)
    end

    return cg
end

include("helpers.jl")
include("gen.jl")
