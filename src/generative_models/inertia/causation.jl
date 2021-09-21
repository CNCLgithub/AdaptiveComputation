function causal_init(gm::GMParams,
                     dm::InertiaModel,
                     gr::Graphics,
                     things::AbstractArray{Thing})
    cg = CausalGraph(SimpleDiGraph())
    ws = init_walls(gm.area_width, gm.area_height)
    nnew = length(ws) + length(things)
    born = [ws; things]
    ch = Dict{ChangeDiff, Any}(
        (:gm => :gm) => gm,
        (:dm => :dm) => dm,
        (:graphics => :graphics) => gr,
    )
    d = Diff(born, Int64[], StaticPath[], ch)
    patch(cg, d)
end

function causal_update(dm::InertiaModel, prev_cg::CausalGraph,
                       initdiff::Diff)::CausalGraph

    # first patch
    #  1. birth/death process
    #  2. static walls, module parameters
    #  3. motion updates for trackers + flow mask propagation
    #
    # currently, walls are static and flow masks are propagated
    # to be using during `graphics_update`
    #
    # this creates a new `CausalGraph` thus breaking reference
    # to the graph at t-1 (for inference safety)
    cg = @>> module_diff merge(initdiff) patch(prev_cg)

    # all future patches are done in place for effeciency
    #
    # compute forces (dynamics)
    @>> cg dynamics_update(dm) patch!(cg)
    # resolve forces into kinematic changes
    @>> cg kinematics_update(dm) patch!(cg)

    # apply graphics changes
    # create the flows and spaces for each element
    # inplace patch:
    #   1. :flow (graphics state w/ memory)
    #   2. :space (current mass matrix)
    gr = get_graphics(cg)
    @>> cg render(gr) patch!(cg)
    # update predictions
    @>> cg predict(gr) patch!(cg)

    cg
end

const module_diff = Diff(StaticPath[
    :gm => :gm,
    :dm => :dm,
    :graphics => :graphics,
])
