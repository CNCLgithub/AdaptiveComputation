using MOT
using Random
Random.seed!(1)


output_dir = "/renders/exp1_intro_movies"
mkpath(output_dir)
k = 120

# loading the motion model
q = Exp0(scene=40)
gm = MOT.load(GMMaskParams, q.gm)
scene_data = load_scene(q.scene, q.dataset_path, gm;
                        generate_masks=false)

gm = GMMaskParams()
# generating new scene data
scene_data = dgp(k, gm, scene_data[:motion];
                 generate_masks=false)

cgs = scene_data[:gt_causal_graphs]
z = map(x->x.pos[3], cgs[1].elements)
for i=1:length(cgs)
    for j=1:length(cgs[i].elements)
        cgs[i].elements[j].pos[3] = z[j]
    end
end

# intro with no labels
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_no_label"),
       stimuli=true)

# intro target designation
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_target_designation"),
       freeze_time=24,
       stimuli=true)

# probes
t = 40
for i=t-2:t+2
    dot = cgs[i+1].elements[1]
    cgs[i+1].elements[1] = Dot(pos = dot.pos,
                                     vel = dot.vel,
                                     probe = true,
                                     radius = dot.radius,
                                     width = dot.width,
                                    height = dot.height)
end
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_probe"),
       freeze_time=24,
       stimuli=true)



# intro full
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_full"),
       highlighted=[1],
       freeze_time=24,
       stimuli=true)
