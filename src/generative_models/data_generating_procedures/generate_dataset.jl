export generate_dataset

function is_min_distance_satisfied(scene_data, min_distance)
    init_dots = scene_data[:gt_causal_graphs][1].elements
    distances = map(x -> map(y -> MOT.dist(x.pos[1:2], y.pos[1:2]), init_dots), init_dots)
    satisfied = map(distance -> distance == 0.0 || distance > min_distance, Iterators.flatten(distances))
    all(satisfied)
end

function generate_dataset(dataset_path, n_scenes, k, gm, motion;
                          min_distance = 50.0)
    jldopen(dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            scene_data = nothing
            while true
                scene_data = dgp(k, gm, motion)
                if is_min_distance_satisfied(scene_data, min_distance)
                    break
                end
            end
            scene = JLD2.Group(file, "$i")
            scene["gm"] = gm
            scene["motion"] = motion

            gt_cgs = scene_data[:gt_causal_graphs]

            # fixing z according to the time 0
            z = map(x->x.pos[3], gt_cgs[1].elements)
            map(cg -> map(i -> cg.elements[i].pos[3] = z[i],
                          collect(1:length(cg.elements))),
                          gt_cgs)
            scene["gt_causal_graphs"] = gt_cgs
        end
    end
end
