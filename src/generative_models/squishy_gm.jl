
@gen function sample_init_polygon(gmh)
    
    #n_dots = @trace(categorical([0.5; fill((1-0.5)/5, 5)]), :n_dots)

    max_dots = 5
    n_dots = @trace(categorical(fill(1/max_dots, max_dots), :n_dots))
    
    x = @trace(uniform(-gmh.init_pos_spread, gmh.init_pos_spread), :x)
    y = @trace(uniform(-gmh.init_pos_spread, gmh.init_pos_spread), :y)
    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)
    

    if n_dots == 1
        pol = UGon([x,y,z], zeros(3))
        dots = Dot[]
        return (pol, dots)
    else
        rot = @trace(uniform(0, 2*pi), :rot)
        r = gmh.polygon_radius
        pol = NGon([x,y,z], rot, zeros(3), 0.0, r, n_dots)

        dots = Vector{Dot}(undef, n_dots)
        for i=1:n_dots
            # creating dots along the polygon
            dot_x = x + r * cos(2*pi*i/n_dots + rot)
            dot_y = y + r * sin(2*pi*i/n_dots + rot)
            
            # sprinkling some noise
            dot_x = @trace(normal(dot_x, 5.0), i => :x)
            dot_y = @trace(normal(dot_y, 5.0), i => :y)

            dots[i] = Dot([dot_x,dot_y,z], [0.0,0.0])
        end

        return (pol, dots)
    end
end


function init_walls(hgm::HGMParams)
    # getting wall points
    wp = @>> Iterators.product((-1,1), (-1,1)) begin
        map(x -> x .* (hgm.area_width/2, hgm.area_height/2))
    end
    
    println(wp[1])
    # getting the walls
    ws = Vector{Wall}(undef, 4)
    ws[1] = Wall(wp[1], wp[2])
    ws[2] = Wall(wp[2], wp[3])
    ws[3] = Wall(wp[3], wp[4])
    ws[4] = Wall(wp[4], wp[1])
    
    println(ws)
    return ws
end

@gen function sample_init_squishy_state(hgm::HGMParams,
                                        dm::SquishyDynamicsModel)

    hgm_trackers = fill(hgm, hgm.n_trackers)
    ws = init_walls(hgm)
    current_state = @trace(Gen.Map(sample_init_polygon)(hgm_trackers), :polygons)
    
    #graph = CausalGraph(trackers, SimpleGraph)
    cg = CausalGraph(SimpleDiGraph())
    for w in walls_idx(dm)
        add_vertex!(cg)
        set_props!(cg, w, :object, ws[w])
    end
    set_prop!(cg, :walls, walls_idx(dm))
    for (poly, verts) in current_state
        add_vertex!(cg)
        poly_v = nv(cg)
        set_prop!(cg, poly_v, :object, poly)

        for v in verts
            add_vertex!(cg)
            vi = nv(cg)
            add_edge!(cg, poly_v, vi)
            set_props!(cg, vi, :object, v)
            set_prop!(cg, Edge(poly_v, vi),
                      :parent, true)
        end
    end

    pmbrfs = RFSElements{Array}(undef, 0)

    if hgm.fmasks
        fmasks = Array{Matrix{Float64}}(undef, hgm.n_trackers, hgm.fmasks_n)
        for i=1:hgm.n_trackers
            for j=1:hgm.fmasks_n
                fmasks[i,j] = zeros(hgm.img_height, hgm.img_width)
            end
        end
        flow_masks = FlowMasks(fmasks,
                               hgm.fmasks_decay_function)
    else
        flow_masks = nothing
    end

    return State(cg, pmbrfs, flow_masks)
end


@gen function squishy_gm_pos_kernel(t::Int,
                                    prev_state::State,
                                    dm::AbstractDynamicsModel,
                                    hgm::HGMParams)
    prev_graph = prev_state.graph
    new_graph = @trace(squishy_update(dm, prev_graph, hgm), :dynamics)
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = State(new_graph, pmbrfs, nothing)
    return new_state
end

@gen function squishy_gm_pos(k::Int,
                             dm::SquishyDynamicsModel,
                             hgm::HGMParams)
    init_state = @trace(sample_init_squishy_state(hgm, dm), :init_state)
    states = @trace(Gen.Unfold(squishy_gm_pos_kernel)(k, init_state, dm, hgm), :kernel)
    result = (init_state, states)
    return result
end

export squishy_gm_pos

