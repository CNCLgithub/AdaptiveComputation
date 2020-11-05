using MOT
using Random
Random.seed!(1)

output_dir = "/renders/isr_inertia_extended_intro_movies"
mkpath(output_dir)
scene = 12
k = 120
dataset_path = "/datasets/exp1_isr_extended.jld2"

scene_data = load_scene(scene, dataset_path, default_gm;
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
pad = 2
for i=t-pad:t+pad
    dot = probed_cgs[i+1].elements[2]
    probed_cgs[i+1].elements[2] = Dot(pos = dot.pos,
                                     vel = dot.vel,
                                     probe = true,
                                     radius = dot.radius,
                                     width = dot.width,
                                    height = dot.height)
end

# intro with no labels
render(default_gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_no_label"),
       stimuli=true)

# intro target designation snap
render(default_gm, 0;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_target_designation_snap"),
       freeze_time=1,
       stimuli=true)
rm(joinpath(output_dir, "intro_target_designation_snap", "002.png"))

# intro target designation
render(default_gm, k;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_target_designation"),
       freeze_time=75,
       stimuli=true)

# intro probe snap
render(default_gm, 1;
       gt_causal_graphs=probed_cgs[t:t+1],
       path=joinpath(output_dir, "intro_probe_snap"),
       freeze_time=0,
       stimuli=true)

# intro probe
render(default_gm, t+10;
       gt_causal_graphs=probed_cgs,
       path=joinpath(output_dir, "intro_probe"),
       freeze_time=24,
       stimuli=true)

# intro td snap
render(default_gm, 0;
       gt_causal_graphs=cgs,
       path=joinpath(output_dir, "intro_query_snap"),
       highlighted=[1],
       freeze_time=1,
       stimuli=true)
rm(joinpath(output_dir, "intro_query_snap", "001.png"))


# doing new probe
probed_cgs = deepcopy(cgs)
t = 45
pad = 2
for i=t-pad:t+pad
    dot = probed_cgs[i+1].elements[2]
    probed_cgs[i+1].elements[2] = Dot(pos = dot.pos,
                                     vel = dot.vel,
                                     probe = true,
                                     radius = dot.radius,
                                     width = dot.width,
                                    height = dot.height)
end


# intro full
render(default_gm, t+10;
       gt_causal_graphs=probed_cgs,
       path=joinpath(output_dir, "intro_full"),
       highlighted=[5],
       freeze_time=24,
       stimuli=true)

