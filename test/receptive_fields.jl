using MOT
using Gen
using Random
using Images, FileIO, PaddedViews
Random.seed!(1)

r_fields_dim = (4, 4)
k = 3
prob_threshold = 0.001
dynamics_model = ISRDynamics()
gm = GMMaskParams(img_width = 800,
                  img_height = 800)

display(gm)

# automatically dividing the img into rectangular receptive_fields
receptive_fields = get_rectangle_receptive_fields(r_fields_dim..., gm)

trace, _ = Gen.generate(gm_receptive_fields, (k, dynamics_model, gm, receptive_fields, prob_threshold))

# rendering normally
(init_state, states) = Gen.get_retval(trace)
gt_causal_graphs = Vector{CausalGraph}(undef, k+1)
gt_causal_graphs[1] = init_state.graph
foreach(t -> gt_causal_graphs[t+1] = states[t].graph, 1:k)
render(gm, k; gt_causal_graphs = gt_causal_graphs)


# looking at sampled masks for receptive fields
function save_img(t, rf, choices, out_dir)
    masks = choices[:kernel => t => :receptive_fields => rf => :masks]
    masks == [] && return

    h, w = size(first(masks))
    masks = map(mask -> PaddedView(1, mask, (1:h+10, 1:w+10), (6:h+5, 6:w+5)), masks)
    for i=1:length(masks)
        fn = joinpath(out_dir, "$(t)_$(rf)_$(i).png")
        save(fn, masks[i])
    end

    or = (x, y) -> x .| y
    aggregate_mask = reduce(or, masks)
    fn = joinpath(out_dir, "$(t)_$(rf).png")
    save(fn, aggregate_mask)
end
    
choices = Gen.get_choices(trace)
out_dir = joinpath("output", "render", "receptive_fields")
rm(out_dir, recursive=true)
mkpath(out_dir)

idxs = CartesianIndices((1:k, 1:r_fields_dim[1]*r_fields_dim[2]))
foreach(idx -> save_img(idx[1], idx[2], choices, out_dir), idxs)
