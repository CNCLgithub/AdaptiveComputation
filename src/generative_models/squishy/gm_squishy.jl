
include("helpers.jl")

@gen function sample_init_polygon(cg::CausalGraph)::Polygon
    @unpack max_vertices, init_pos_spread, dist_pol_verts = (get_gm(cg))
    n_dots = @trace(Gen.categorical(fill(1.0/max_vertices, max_vertices)), :n_dots)
    
    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)
    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    #vel = @trace(broadcasted_normal(zeros(2), 10.0), :vel)
    vel = zeros(2)

    if n_dots == 1
        pol = UGon([x,y,z], vel)
        dots = Dot[Dot([x,y,z], zeros(2))]
        return (pol, dots)
    else
        rot = @trace(uniform(0, 2*pi), :rot)
        r = d_to_r_pol(dist_pol_verts, n_dots)
        #avel = @trace(normal(0.0, 0.05), :avel)
        avel = 0.0
        pol = NGon([x,y,z], rot, vel, avel, r, n_dots)

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

@gen function sample_init_squishy_cg(cg::CausalGraph)
    @unpack n_trackers = (get_gm(cg))
    cgs = fill(cg, n_trackers)
    init_trackers = @trace(Gen.Map(sample_init_polygon)(cgs), :polygons)
    ensemble = UniformEnsemble(cg)

    dynamics_init!(cg, [init_trackers; ensemble])
    graphics_init!(cg)

    return State(cg, pmbrfs, flow_masks)
end

@gen function squishy_gm_pos_kernel(t::Int, prev_cg::CausalGraph)
    cg = @trace(squishy_update(prev_cg), :dynamics) # deepcopy inside
    return cg
end

@gen function squishy_gm_pos(k::Int,
                             cg::CausalGraph)
    init_cg = @trace(sample_init_squishy_cg(cg), :init_state)
    cgs = @trace(Gen.Unfold(squishy_gm_pos_kernel)(k, init_cg), :kernel)
    result = (init_state, states)
    return result
end

export squishy_gm_pos

