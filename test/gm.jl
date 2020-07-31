using MOT
using Gen
using Profile
using StatProfilerHTML

# _lm = Dict(:tracker_positions => extract_tracker_positions,
#            :assignments => extract_assignments)
# latent_map = LatentMap(_lm)

gm_params = GMMaskParams(img_width = 100,
                         img_height = 100)

# generating initial positions and masks (observations)

dataset_path = "datasets/exp_0.h5"
init_positions, masks, motion, positions = load_exp0_trial(1,
                                                           gm_params,
                                                           dataset_path)

# initial observations based on init_positions
# model knows where trackers start off
constraints = Gen.choicemap()
for i=1:size(init_positions, 1)
    addr = :init_state => :trackers => i => :x
    constraints[addr] = init_positions[i,1]
    addr = :init_state => :trackers => i => :y
    constraints[addr] = init_positions[i,2]
end

# compiling further observations for the model
args = (1, motion, gm_params)
obs = Gen.choicemap()
obs[:states => 1 => :masks] = masks[1]

@time (trace, ls) = Gen.generate(gm_masks_static, (0, motion, gm_params), constraints)

@time (tr, ls, _, _) = Gen.update(trace, (1, motion, gm_params), (UnknownChange(),), obs)

Profile.clear()
Profile.init(;n = 10000000, delay = 1E-5)
@profilehtml (tr, ls, _, _) = Gen.update(trace, (1, motion, gm_params), (UnknownChange(),), obs);
@profilehtml (tr, ls, _, _) = Gen.update(trace, (1, motion, gm_params), (UnknownChange(),), obs);
