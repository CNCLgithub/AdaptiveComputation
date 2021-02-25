using MOT
using Gen
using Random
using Images
Random.seed!(6)

k = 100
dm = SquishyDynamicsModel()
cm = Gen.choicemap()
structure = [4, 4, 4, 4]
for (i, s) in enumerate(structure)
    cm[:init_state => :polygons => i => :n_dots] = s
end

targets = Bool[fill(1, 8); fill(0, 8)]

hgm = HGMParams(n_trackers = length(structure),
                distractor_rate = 0.0,
                targets = [1, 1, 0, 0])
scene_data = nothing
tries = 0
while true
    global tries += 1
    print("tries $tries \r")
    global scene_data = dgp(k, hgm, dm;
                     generate_masks=false,
                     cm=cm)
    md = is_min_distance_satisfied(scene_data, 50.0)
    #md = true
    di = are_dots_inside(scene_data, hgm)
    md && di && break
end

render(hgm, k;
       gt_causal_graphs=scene_data[:gt_causal_graphs],
       highlighted_start=targets,
       path=joinpath("/renders", "squishy"),
       freeze_time=12,
       show_forces=false,
       show_polygons=false,
       show_polygon_centroids=false)

# trace, _ = Gen.generate(hgm_mask, (k, motion, hgm), cm)
# #display(get_choices(trace))

# init_state, states = Gen.get_retval(trace)

# gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
# gt_causal_graphs[1] = init_state.graph
# gt_causal_graphs[2:end] = map(x->x.graph, states)

# masks = get_masks(gt_causal_graphs, hgm)
# full_imgs = get_full_imgs(masks)

# for (i, img) in enumerate(full_imgs)
    # save("imgs/$i.png", img)
# end

