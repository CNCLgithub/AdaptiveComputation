using MOT

dataset_path = joinpath("output", "datasets", "exp3_polygons_v3.jld2")
scene_data = MOT.load_scene(9, dataset_path, HGMParams();
                            generate_masks=false)

gm = scene_data[:gm]
k = length(scene_data[:gt_causal_graphs])
gt_cgs = scene_data[:gt_causal_graphs]

render(gm, k;
       gt_causal_graphs=gt_cgs,
       path=joinpath("/renders", "velocity"),
       freeze_time=0,
       show_velocity=true,
       show_forces=false,
       show_polygons=false,
       show_polygon_centroids=false)
