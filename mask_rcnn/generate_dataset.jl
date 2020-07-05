using MOT
using Images
using JSON
using Random

dataset_path = joinpath("output", "datasets", "mask_rcnn")
mkpath(dataset_path)
#n_batches = 200
n_examples = 10

spring = LinRange(0.0005, 0.0025, 10)
sigma_w = LinRange(1.0, 2.5, 10)
n_dots = collect(8:16)

batches = Iterators.product(spring, sigma_w, n_dots)
n_batches = length(batches)

# most of the action is later, so skipping first timesteps
start_timestep = 61
n_timesteps = 120

# dumping some information about the dataset to a json
dict = Dict("n_batches" => n_batches,
            "n_examples" => n_examples)
data = JSON.json(dict)
json_path = joinpath(dataset_path, "info.json")
open(json_path, "w") do f
    write(f, data)
end

target_pngs_path = joinpath(dataset_path, "target_pngs")
mkpath(target_pngs_path)

for (batch, elements) in enumerate(batches)
    println("batch $batch")
    
    if batch > 10
        error()
    end
    # sample some timesteps from the back of the trial
    timesteps = collect(start_timestep:n_timesteps)
    perm = randperm(length(timesteps))
    timesteps = timesteps[perm][1:n_examples]
    println("sampled timesteps: $timesteps")
    

    """
    q = Exp0(trial=batch)
    gm = MOT.load(GMMaskParams, q.gm)
    init_positions, masks, motion, positions = load_exp0_trial(batch, gm, q.dataset_path,
                                                               generate_masks=false)
    """
    
    spring, sigma_w, n_trackers = elements
    gm = GMMaskParams(n_trackers = n_trackers,
                      distractor_rate = 0,
                      init_pos_spread = 200)
    motion = BrownianDynamicsModel(spring=spring, sigma_x=sigma_w, sigma_y=sigma_w)
    init_positions, init_vels, masks, positions = dgp(n_timesteps, gm, motion)


    # replacing z with y for a fixed ordering based on y
    for t=1:length(positions)
        for i=1:size(positions[t],1)
            positions[t][i,3] = positions[t][i,2]
        end
    end

    # rendering the input images of the trial
    # and saving to trial path
    render(positions[timesteps], gm, stimuli=true,
           dir=joinpath(dataset_path, "input_pngs"),
           prefix="$(lpad(batch,3,"0"))_")
    
    masks = get_masks(positions[timesteps], gm.dot_radius,
                      gm.area_height, gm.area_width,
                      gm.area_height, gm.area_height,
                      background=true)
    
    for t=1:length(masks)
        index = 1
        for i=1:length(masks[t])
            if !any(masks[t][i])
                println("ZERO MASK")
                println("$batch $t $i")
                continue
            end
            fname = "$(lpad(batch,3,"0"))_$(lpad(t,3,"0"))_$(lpad(index,3,"0")).png"
            save(joinpath(target_pngs_path, fname), masks[t][i])
            index += 1
        end
    end
    println()
end

