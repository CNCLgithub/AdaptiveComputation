function get_repulsion_from_wall(distance, wall_repulsion, pos, gm_params)
    # repulsion from walls
    walls = Matrix{Float64}(undef, 4, 3)
    walls[1,:] = [gm_params.area_width/2, pos[2], pos[3]]
    walls[2,:] = [pos[1], gm_params.area_height/2, pos[3]]
    walls[3,:] = [-gm_params.area_width/2, pos[2], pos[3]]
    walls[4,:] = [pos[1], -gm_params.area_height/2, pos[3]]

    force = zeros(3)
    for j = 1:4
        v = pos - walls[j,:]
        #(norm(v) > min_distance) && continue
        absolute_force = wall_repulsion*exp(-(v[1]^2 + v[2]^2)/(distance^2))
        force .+= absolute_force * v/norm(v)
    end
    return force
end


function get_repulsion_object_to_object(distance, repulsion, pos, other_pos)
    force = zeros(3)
    for j = 1:length(other_pos)
        v = pos - other_pos[j]
        #(norm(v) > distance) && continue
        absolute_force = repulsion*exp(-(v[1]^2 + v[2]^2)/(distance^2))
        force .+= absolute_force * v/norm(v)
    end
    return force
end

function get_repulsion_force_dots(model, objects, gm_params)
    
    n = length(objects)
    #dots_inds = @>> 1:n filter(i -> objects[i] isa Dot)
    rep_forces = fill(zeros(2), n)
    positions = map(d->d.pos, objects)

    #for i = dots_inds
    for i = 1:n
        dot = objects[i]
        
        other_pos = positions[map(j -> i != j, 1:n)]
        dot_applied_force = get_repulsion_object_to_object(model.distance, model.dot_repulsion, dot.pos, other_pos)
        wall_applied_force = get_repulsion_from_wall(model.distance, model.wall_repulsion, dot.pos, gm_params)
        
        println("WALLLLLL!!!")
        println(wall_applied_force)
        println("DOT!!!!!!")
        println(dot_applied_force)
        println()

        rep_forces[i] = dot_applied_force[1:2]+wall_applied_force[1:2]
    end
    
    rep_forces
end

function isr_repulsion_step(cg::CausalGraph)::CausalGraph
    dm = get_dm(cg)
    dots = get_objects(cg, Dot)
    gm = get_gm(cg)

    rep_forces = get_repulsion_force_dots(dm, dots, gm)
    n = length(dots)

    for i = 1:n
        vel = dots[i].vel
        if sum(vel) != 0
            vel *= dm.vel/norm(vel)
        end
        vel *= dm.rep_inertia
        vel += (1.0-dm.rep_inertia)*(rep_forces[i])
        dots[i] = Dot(dots[i].pos, vel)
    end
    
    return dots
end


function dynamics_init!(dm::InertiaModel, gm::GMParams,
                        cg::CausalGraph, things)

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

function dynamics_update!(dm::InertiaModel,
                          cg::CausalGraph,
                          things)
    vs = get_object_verts(cg, Dot)

    for (i, thing) in enumerate(things)
        set_prop!(cg, vs[i], :object, thing)
    end

    return cg
end

walls_idx(dm::InertiaModel) = collect(1:4)

function get_walls(cg::CausalGraph, dm::InertiaModel)
    @>> walls_idx(dm) begin
        map(v -> get_prop(cg, v, :object))
    end
end
