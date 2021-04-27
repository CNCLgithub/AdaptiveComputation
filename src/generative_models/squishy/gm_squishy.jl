
include("helpers.jl")

@gen function sample_init_polygon(hgm)
    
    #n_dots = @trace(categorical([0.5; fill((1-0.5)/5, 5)]), :n_dots)

    @unpack max_vertices, init_pos_spread, dist_pol_verts = hgm
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


@gen function sample_init_squishy_state(hgm::HGMParams,
                                        dm::SquishyDynamicsModel)

    hgm_trackers = fill(hgm, hgm.n_trackers)
    ws = init_walls(hgm)
    current_state = @trace(Gen.Map(sample_init_polygon)(hgm_trackers), :polygons)
    
    cg = process_temp_state(current_state, hgm, dm)
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
    new_cg = @trace(squishy_update(dm, prev_state.cg, hgm), :dynamics)
    pmbrfs = prev_state.rfs # pass along this reference for effeciency
    new_state = State(new_cg, pmbrfs, nothing)
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

