using MOT
using Gen
using Random
using Images
Random.seed!(2)

k = 120
motion = HGMDynamicsModel()
cm = Gen.choicemap()
cm[:init_state => :trackers => 1 => :polygon] = true
cm[:init_state => :trackers => 1 => :n_dots] = 4
cm[:init_state => :trackers => 2 => :polygon] = true
cm[:init_state => :trackers => 2 => :n_dots] = 3
cm[:init_state => :trackers => 3 => :polygon] = false
targets = Bool[1, 1, 1, 0, 1, 0, 0, 0]

hgm = HGMParams(n_trackers = 3,
                distractor_rate = 0.0,
                targets = [1, 1, 0, 0])

scene_data = dgp(k, hgm, motion;
                 generate_masks=false,
                 cm=cm)

render(hgm, k;
       gt_causal_graphs=scene_data[:gt_causal_graphs],
       highlighted=collect(1:8)[targets],
       freeze_time=2,
       stimuli=false)

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
