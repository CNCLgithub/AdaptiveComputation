function init_cg_from_trackers(cg::CausalGraph, trackers::AbstractArray)
    @>> trackers begin
        collect(Thing)
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
