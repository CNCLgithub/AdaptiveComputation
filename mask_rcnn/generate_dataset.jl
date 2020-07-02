using MOT
using Images
using JSON

# rolling through
#springs = LinRange(0.001, 0.003, 10)
#sigma_ws = LinRange(0.5, 2.5, 10)
#n_dots = collect(1:16)

dataset_path = joinpath("output", "datasets", "mask_rcnn")
n_trials = 1
n_timesteps = 120

# note that 
target_pngs_path = joinpath(dataset_path, "target_pngs")
mkpath(target_pngs_path)

for trial=1:n_trials
    gm = GMMaskParams(n_trackers = 4, distractor_rate = 4)
    motion = BrownianDynamicsModel()
    init_positions, init_vels, masks, positions = dgp(n_timesteps, gm, motion)

    # rendering the input images of the trial
    # and saving to trial path
    render(positions, gm, stimuli=true,
           #dir=joinpath(trial_path, "input_pngs"),
           dir=joinpath(dataset_path, "input_pngs"),
           prefix="$(lpad(trial,3,"0"))_")
    
    masks = get_masks(positions, gm.dot_radius,
                      gm.area_height, gm.area_width,
                      gm.area_height, gm.area_height,
                      background=true)
    
    for t=1:length(masks)
        for i=1:length(masks[t])
            fname = "$(lpad(trial,3,"0"))_$(lpad(t,3,"0"))_$(lpad(i,3,"0")).png"
            save(joinpath(target_pngs_path, fname), masks[t][i])
        end
    end
end

