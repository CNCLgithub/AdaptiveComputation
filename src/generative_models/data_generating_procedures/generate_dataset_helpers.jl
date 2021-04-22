function init_constraint_from_cg(cg::MOT.CausalGraph, old_cm::ChoiceMap)
    cm = choicemap()

    # getting general initial structure
    submap = Gen.get_submap(old_cm, :init_state)
    
    # need to put the polygons into fastforwarded places
    pol_verts = @>> MOT.get_object_verts(cg, MOT.Polygon)
    for (i, pol_vert) in enumerate(pol_verts)
        cm[:init_state => :polygons => i => :n_dots] = submap[:polygons => i => :n_dots]
        pol = get_prop(cg, pol_vert, :object)
        pos = MOT.get_pos(pol)
        cm[:init_state => :polygons => i => :x] = pos[1]
        cm[:init_state => :polygons => i => :y] = pos[2]
        cm[:init_state => :polygons => i => :z] = pos[3]
        if pol isa NGon
            cm[:init_state => :polygons => i => :rot] = pol.rot
        end

        dot_verts = LightGraphs.vertices(cg, pol_vert)
        for (j, dot_vert) in enumerate(dot_verts)
            dot = get_prop(cg, dot_vert, :object)
            pos = MOT.get_pos(dot)
            cm[:init_state => :polygons => i => j => :x] = pos[1]
            cm[:init_state => :polygons => i => j => :y] = pos[2]
        end
    end
    
    cm
end

function forward_scene_data!(scene_data, timestep)
    scene_data[:gt_causal_graphs] = scene_data[:gt_causal_graphs][timestep:end]
    if !isnothing(scene_data[:masks])
        scene_data[:masks] = scene_data[:masks][timestep:end]
    end
end

function are_dots_inside(scene_data, gm)
    d = gm.dot_radius
    xmin, xmax = -gm.area_width/2 + d, gm.area_width/2 - d
    ymin, ymax = -gm.area_height/2 + d, gm.area_width/2 - d
    
    cg = first(scene_data[:gt_causal_graphs])
    dots = get_objects(cg, Dot)
    positions = @>> dots map(d -> d.pos)

    satisfied = map(i ->
                    positions[i][1] > xmin &&
                    positions[i][1] < xmax &&
                    positions[i][2] > ymin &&
                    positions[i][2] < ymax,
                    1:length(dots))
    
    all(satisfied)    
end

function is_min_distance_satisfied(scene_data, min_distance)
    cg = first(scene_data[:gt_causal_graphs])
    
    objects = collect(filter_vertices(cg, :object))
    positions = @>> objects begin
        map(v -> get_prop(cg, v, :object))
        filter(obj -> obj isa Dot)
        map(get_pos)
    end

    n_objects = length(positions)

    distances_idxs = Iterators.product(1:n_objects, 1:n_objects)
    distances = @>> distances_idxs map(xy -> MOT.dist(positions[xy[1]][1:2], positions[xy[2]][1:2]))
    distances = @>> distances map(x -> x == 0.0 ? Inf : x)
    println("minimum distance: $(minimum(distances))")

    satisfied = @>> distances map(d -> d > min_distance)
    all(satisfied)
end
