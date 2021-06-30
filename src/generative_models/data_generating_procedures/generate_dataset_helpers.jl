
"""
Returns a choicemap with dots in positions according to the
causal graph
"""
function init_constraint_from_cg(cg::MOT.CausalGraph)
    dots = get_objects(cg, Dot)
    cm = choicemap()
    
    for (i, d) in enumerate(dots)
        cm[:init_state => :trackers => i => :x] = d.pos[1]
        cm[:init_state => :trackers => i => :y] = d.pos[2]
    end

    return cm
end

"""
    Returns true iff the dots are inside the area boundaries
    for every timestep
"""
function are_dots_inside(cgs, gm)
    d = gm.dot_radius
    xmin, xmax = -gm.area_width/2 + d, gm.area_width/2 - d
    ymin, ymax = -gm.area_height/2 + d, gm.area_width/2 - d
    
    for t=1:length(cgs)
        pos = @>> get_objects(cgs[t], Dot) map(get_pos)
        satisfied = map(i ->
                        pos[i][1] > xmin &&
                        pos[i][1] < xmax &&
                        pos[i][2] > ymin &&
                        pos[i][2] < ymax,
                        1:length(pos))
        !all(satisfied) && return false
    end

    return true 
end

# Returns true iff minimum distance between the dots is satisfied.
function is_min_distance_satisfied(first_cg::CausalGraph, min_distance::Float64)
    positions = @>> get_objects(first_cg, Dot) map(get_pos)
    n_objects = length(positions)

    distances_idxs = Iterators.product(1:n_objects, 1:n_objects)
    satisfied = @>> distances_idxs begin
        map(xy -> MOT.dist(positions[xy[1]][1:2], positions[xy[2]][1:2]))
        map(x -> x == 0.0 ? Inf : x) # if the same object, satisfied
        map(x -> x > min_distance)
        all
    end
end
