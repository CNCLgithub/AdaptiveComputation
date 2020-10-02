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

probed_cgs = deepcopy(cgs)
t = 40
pad = 1
for i=t-pad:t+pad
    dot = probed_cgs[i+1].elements[1]
    probed_cgs[i+1].elements[1] = Dot(pos = dot.pos,
                                     vel = dot.vel,
                                     probe = true,
                                     radius = dot.radius,
                                     width = dot.width,
                                    height = dot.height)
end

# intro with no labels
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_no_label"),
       stimuli=true)

# intro target designation snap
render(gm, 0;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_target_designation_snap"),
       freeze_time=1,
       stimuli=true)
rm(joinpath(output_dir, "intro_target_designation_snap", "002.png"))

# intro target designation
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_target_designation"),
       freeze_time=75,
       stimuli=true)

# intro probe snap
render(gm, 1;
       gt_causal_graphs=probed_cgs[t:t+1],
       path=joinpath(output_dir, "intro_probe_snap"),
       freeze_time=0,
       stimuli=true)

# intro probe
render(gm, k;
       gt_causal_graphs=probed_cgs,
       path=joinpath(output_dir, "intro_probe"),
       freeze_time=24,
       stimuli=true)

# intro td snap
render(gm, 0;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_td_snap"),
       highlighted=[1],
       freeze_time=1,
       stimuli=true)
rm(joinpath(output_dir, "intro_td_snap", "001.png"))

# intro pr snap
render(gm, 0;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_pr_snap"),
       highlighted=Int[],
       freeze_time=1,
       stimuli=true)
rm(joinpath(output_dir, "intro_pr_snap", "001.png"))

# intro td
render(gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_td"),
       highlighted=[1],
       freeze_time=24,
       stimuli=true)


# doing new probe
probed_cgs = deepcopy(cgs)
t = 45
pad = 1
for i=t-pad:t+pad
    dot = probed_cgs[i+1].elements[2]
    probed_cgs[i+1].elements[2] = Dot(pos = dot.pos,
                                     vel = dot.vel,
                                     probe = true,
                                     radius = dot.radius,
                                     width = dot.width,
                                    height = dot.height)
end

# intro pr
render(gm, k;
       gt_causal_graphs=probed_cgs,
       path=joinpath(output_dir, "intro_pr"),
       highlighted=Int[],
       freeze_time=24,
       stimuli=true)
