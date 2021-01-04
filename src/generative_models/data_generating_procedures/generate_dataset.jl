export generate_dataset, is_min_distance_satisfied

function is_min_distance_satisfied(scene_data, min_distance)
    init_dots = scene_data[:gt_causal_graphs][1].elements
    distances = map(x -> map(y -> MOT.dist(x.pos[1:2], y.pos[1:2]), init_dots), init_dots)
    satisfied = map(distance -> distance == 0.0 || distance > min_distance, Iterators.flatten(distances))
    all(satisfied)
end

function generate_dataset(dataset_path, n_scenes, k, gm, motion;
                          min_distance = 50.0,
                          cms::Union{Nothing, Vector{ChoiceMap}} = nothing,
                          aux_data::Union{Nothing, Vector{Any}} = nothing)

    jldopen(dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            print("generating scene $i/$n_scenes \r")
            scene_data = nothing

            # if no choicemaps, then create an empty one
            cm = isnothing(cms) ? choicemap() : cms[i]

            while true
                scene_data = dgp(k, gm, motion;
                                 generate_masks=false,
                                 cm=cm)
                if is_min_distance_satisfied(scene_data, min_distance)
                    break
                end
            end

            scene = JLD2.Group(file, "$i")
            scene["gm"] = gm
            scene["motion"] = motion
            scene["aux_data"] = isnothing(aux_data) ? nothing : aux_data[i]

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
