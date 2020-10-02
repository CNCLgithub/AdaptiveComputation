using MOT

include("../scripts/stimuli/exp0_probes.jl")

q = Exp0(scene=1)
gm = MOT.load(GMMaskParams, q.gm)
scene_data = load_scene(q.scene, q.dataset_path, gm;
                        generate_masks=false)

cgs = scene_data[:gt_causal_graphs]
render(gm, q.k;
       gt_causal_graphs=cgs,
       path="testing_rotation/not_rotated")

rotated_cgs = rotate(cgs, pi/2)

render(gm, q.k;
       gt_causal_graphs=rotated_cgs,
       path="testing_rotation/rotated")
