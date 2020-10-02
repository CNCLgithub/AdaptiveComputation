export generate_dataset

function generate_dataset(dataset_path, n_scenes, k, gm, motion)
    jldopen(dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            scene_data = dgp(k, gm, motion;
                             generate_masks=false)

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
