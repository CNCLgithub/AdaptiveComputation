using MOT

dataset_path = joinpath("output", "datasets", "exp3_polygons_v3.jld2")
scene_data = MOT.load_scene(9, dataset_path, HGMParams();
                            generate_masks=false)
r_fields_dim = (5, 5)
gm = scene_data[:gm]
k = length(scene_data[:gt_causal_graphs])
gt_cgs = scene_data[:gt_causal_graphs]
targets = scene_data[:aux_data][:targets]

render(gm, k;
       gt_causal_graphs=gt_cgs,
       highlighted_start=targets,
       path=joinpath("/renders", "receptive_fields"),
       freeze_time=24,
       show_forces=false,
       show_polygons=false,
       show_polygon_centroids=false,
       receptive_fields=r_fields_dim,
       receptive_fields_overlap=5)

