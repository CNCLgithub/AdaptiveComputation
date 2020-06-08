using MOT
using Gen
using Gen_Compose

#using FileIO
using HDF5
using Random
using Profile

using Images
using StatProfilerHTML


# quick function to extract points from the whole trace
# (for testing state proposals)
function extract_points(trace)
    T, params = Gen.get_args(trace)
    xy = Array{Float64}(undef, T, params.num_trackers, 3)
    current_xy = Array{Float64}(undef, params.num_trackers, 3)

    for i=1:params.num_trackers
        current_xy[i,1] = trace[:init_state => :init_trackers => i => :x]
        current_xy[i,2] = trace[:init_state => :init_trackers => i => :y]
        current_xy[i,3] = trace[:init_state => :init_trackers => i => :z]
    end

    for t=1:T
        for i=1:params.num_trackers
            vx = trace[:states => t => :trackers => i => :vx]
            vy = trace[:states => t => :trackers => i => :vy]
            current_xy[i,1:2] += [vx, vy] 
        end
        xy[t,:,:] = current_xy
    end

    return xy
end


function generate_masks(T, params)
    println("generating data...")
    args = (T, params)

    constraints = Gen.choicemap()

    trace, _ = Gen.generate(gm_masks_static, args, constraints)
    choices = Gen.get_choices(trace)

    full_imgs = get_full_imgs(T, choices, params)

    xy = extract_points(trace)

    return choices, full_imgs, xy
end


# generating masks and relevant init choices from exp0 trial
function generate_masks_exp0(trial, T, num_trackers, num_distractors_rate)
    choices = Gen.choicemap()
    obs, avg_vel, dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v = load_from_file("datasets/exp_0.h5", trial)
    
    position_noise = 2.0
    depth_noise = 0.5

    rejuv_smoothness = 1.005
    max_rejuv = 15

    area_width = 800
    area_height = 800

    img_width = 200
    img_height = 200
    dot_radius = 20.0

    attended_trackers = fill([], T)

	params = Params(inertia, spring, sigma_w, sigma_v,
                    num_trackers, num_distractors_rate,
                    rejuv_smoothness,
                    max_rejuv,
                    area_width,
                    area_height,
                    img_width,
                    img_height,
                    dot_radius,
                    attended_trackers)
    
    for i=1:params.num_trackers
        choices[:init_state => :init_trackers => i => :x] = init_dots[i,1]
        choices[:init_state => :init_trackers => i => :y] = init_dots[i,2]
    end
    
    for t=1:T
        points = obs[t]
        perm = randperm(length(points))
        # permuting the points for some random depth ordering
        points = points[perm]
        masks = []

        img_so_far = BitArray{2}(undef, params.img_height, params.img_width)
        img_so_far .= false

        for point in points
            mask = draw_mask(point, img_so_far, params)
            push!(masks, mask)
            img_so_far .|= mask
        end
    
        # permuting them back so that target masks are always [1,2,3,4]
        masks = masks[invperm(perm)]
        choices[:states => t => :masks] = masks
    end
    
    full_imgs = get_full_imgs(T, choices, params)

    return choices, full_imgs, params, obs
end


function run_inference(choices, params, T, num_particles)

    println("preparing for inference...")

    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                               ))

    
    # initial observations based on choices
    # model knows where trackers start off
    init_obs = Gen.choicemap()
    for i=1:params.num_trackers
        addr = :init_state => :init_trackers => i => :x
        init_obs[addr] = choices[addr]
        addr = :init_state => :init_trackers => i => :y
        init_obs[addr] = choices[addr]
    end
    
    # compiling further observations for the model
    args = [(t, params) for t in 1:T]
    observations = Vector{Gen.ChoiceMap}(undef, T)
    for t = 1:T
        cm = Gen.choicemap()
        cm[:states => t => :masks] = choices[:states => t => :masks]
        observations[t] = cm
    end
    
    query = Gen_Compose.SequentialQuery(latent_map, #bogus for now
                                        gm_masks_static,
                                        (0, params),
                                        init_obs,
                                        args,
                                        observations)
    
    procedure = PopParticleFilter(num_particles,
                                            num_particles/2, # ESS is in terms of effective particle count, not fraction
                                            nothing,
                                            tuple(),
                                            rejuvenate_state!, # rejuvenation
                                            retrieve_td_confusability, # population statistic
                                            early_stopping_td_confusability, # stopping criteria
                                            params.max_rejuv,
                                            3,
                                            true)
    
    println("running inference...")
    results = sequential_monte_carlo(procedure, query,
                                     buffer_size = T,
                                     path = nothing)
    
    return results
end


function main()
    trial = 42

    Random.seed!(2)
   
    T = 120
    num_trackers = 4
    num_distractors_rate = 4
    num_particles = 10

    
    # generating masks from exp0 (also get all params from that model)
    choices, full_imgs, params, obs = generate_masks_exp0(trial, T, num_trackers, num_distractors_rate)
    
    # if you want just stimuli
    just_stimuli = false
    if just_stimuli
        overlay(obs, params.num_trackers; folder_name="stimuli", stimuli=true, highlighted=[1,2,3,4])
        return
    end
    
    # you can also generate masks from the generative model itself
    masks_from_gm = true
    if masks_from_gm
        # changing some parameter
        params.inertia = 0.95
        params.sigma_w = 0.5
        params.spring = 0.0005

        choices, full_imgs, _ = generate_masks(T, params)
    end
    
    results = run_inference(choices, params, T, num_particles)

    extracted = extract_chain(results)
    xy = extracted["unweighted"][:tracker_positions]
    
    # extracting attended in a nicer form
    attended = zeros(T, num_trackers)
    rej_moves = zeros(Int, T)
    for t=1:T
        att = params.attended_trackers[t]
        for a in att
           attended[t,a] += 1 
           rej_moves[t] += 1
        end
    end
    attended /= params.max_rejuv*num_particles
    rej_moves /= num_particles
    
    plot_rejuvenation(rej_moves)
    plot_xy(xy)
    
    # this is visualizing what the observations look like (and inferred state too)
    # you can find images under full_imgs/
    visualize(xy, full_imgs, params, T, num_particles)

    # this is vizualizing in the old form, also showing where the model is attending
    # you can find images under render/
    overlay(obs[1:T], params.num_trackers; pf_xy=xy, stimuli=false, highlighted=[1,2,3,4], attended=attended)

    return extracted
end

main()
