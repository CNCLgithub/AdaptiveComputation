
@gen (static) function vert_step(dm::SquishyDynamicsModel,
                                 cg::CausalGraph, poly::Polygon,
                                 v::Int64)
    dot = get_prop(cg, v, :object)
    dv = @trace(broadcasted_normal(poly.vel, dm.vert_sigma), :dv)
    new_vert = update(dot, force(cg, v), dv)
    return new_vert
end


@gen (static) function poly_step(dm::SquishyDynamicsModel,
                                 cg::CausalGraph, v::Int64)
    rep = force(cg, v)
    object = get_prop(cg, v, :object)

    # centroid
    dv = @trace(broadcasted_normal(dm.inertia, dm.sigma), :dv)
    dav = @trace(normal(1.0, dm.pol_ang_vel_sigma),
                 :dav)
    new_p = update(object, rep, dv, dav)

    # vertices
    vs = vertices(cg, v)
    dms, cgs, ps = vert_step_args(dm, cg, new_p)
    new_verts = @trace(Map(vert_step)(dms, cgs, ps, vs),
                       :vertices)
    t = (new_p, new_verts)
    return t
end

@gen (static) function squishy_update(dm::SquishyDynamicsModel,
                                      cg::CausalGraph,
                                      hgm)
    # first update polygons
    dms, cgs, plygs = poly_step_args(dm, cg)
    new_state_temp = @trace(Map(poly_step)(dms, cgs, plygs), :polygons)
    new_cg = process_temp_state(new_state_temp, cg, dm)
    return new_cg
end
