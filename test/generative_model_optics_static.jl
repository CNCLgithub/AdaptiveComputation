using MOT
using Gen
using Gen_Compose
using FileIO
using HDF5
using Random

using Images, ImageDraw

using StatProfilerHTML

include("stress_tests/stress_test_0.jl")


# x or y
function img_union(x,y)
    return x == 1.0 || y == 1.0 ? 1.0 : 0.0
end
function draw_image(masks, params)
    img = AbstractArray{Float64}
    img = zeros(Gray, params.img_height, params.img_width)

    for mask in collect(masks)
        img = img_union.(img, 1.0 .- mask)
    end

    return 1.0 .- img
end


function generate_data(T, num_particles)
    params = deepcopy(default_params)
    args = (T, params)
    trace = Gen.simulate(generative_model_masks_static, args)
    choices = Gen.get_choices(trace)
    results = Gen.get_retval(trace)
   
    full_imgs = []

    """ 
    for t=1:T
        #masks = choices[:states => t => :masks]
        #img = draw_image(masks, default_params)
        #push!(full_imgs, img)
    end
    """

    return choices, full_imgs
end

function run_inference(obs, init_dots, params, T, num_particles)
    println()
    println("INFERENCE TIME WOWOWOOWOWOWOWO!!!!")
    println()

    # inference with observations from the generative model
    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                                #:d_centroid_positions => extract_d_centroid_positions
                               ))

    
    init_obs = Gen.choicemap()
    for i=1:params.num_trackers
        addr = :init_state => :init_trackers => i => :x
        init_obs[addr] = init_dots[i,1] #choices[addr]
        addr = :init_state => :init_trackers => i => :y
        println(init_dots)
        init_obs[addr] = init_dots[i,2] #choices[addr]
    end

    """
        addr = :init_state => :init_trackers => 1 => :x
        init_obs[addr] = init_dots[4,1] #choices[addr]
        addr = :init_state => :init_trackers => 1 => :y
        init_obs[addr] = init_dots[4,2] #choices[addr]
    """

    
    args = [(t, params) for t in 1:T]
    observations = Vector{Gen.ChoiceMap}(undef, T)
    for t = 1:T
        cm = Gen.choicemap()
        #cm[:states => t => :masks] = choices[:states => t => :masks]
        cm[:states => t => :optics] = obs[t] #choices[:states => t => :optics]
        observations[t] = cm
    end

    query = Gen_Compose.SequentialQuery(latent_map, #bogus for now
                                        generative_model_masks_static,
                                        (0, params),
                                        init_obs,
                                        args,
                                        observations)
    do_nothing() = nothing

    procedure = PopParticleFilter(num_particles,
                                            num_particles/4, # ESS is in terms of effective particle count, not fraction
                                            nothing,
                                            tuple(),
                                            rejuvenate_state!, # rejuvenation
                                            retrieve_td_confusability, # population statistic
                                            early_stopping_td_confusability, # stopping criteria
                                            params.max_rejuv,
                                            3,
                                            true)

    results = sequential_monte_carlo(procedure, query,
                                     buffer_size = T,
                                     path = nothing)
    
    return results
end

function visualize(xy, full_imgs, params, T, num_particles)
    # just some non file magic kind of thing
    
    for t=1:length(full_imgs)
        img = AbstractArray{Float64}
        img = Gray.(full_imgs[t])

        for p=1:num_particles
            for i=1:size(xy,3)
                x = xy[t,p,i,1]
                y = xy[t,p,i,2]
                x = round(Int, x+params.img_width/2) 
                y = round(Int, y+params.img_height/2)
                circle = CirclePointRadius(x, y, 7.0)
                draw!(img, circle, Gray(0.0))
                circle = CirclePointRadius(x, y, 5.0)
                draw!(img, circle, Gray(1.0))
                circle = CirclePointRadius(x, y, 3.0)
                draw!(img, circle, Gray(0.0))
                circle = CirclePointRadius(x, y, 1.0)
                draw!(img, circle, Gray(1.0))
            end
        end

        # drawing distractor centroids
        for i=1:size(dxy,2)
            x = dxy[t,i,1]
            y = dxy[t,i,2]
            x = round(Int, x+params.img_width/2) 
            y = round(Int, y+params.img_height/2)
            circle = CirclePointRadius(x, y, 10.0)
            draw!(img, circle, Gray(0.0))
        end
        save("image_$(lpad(t, 3, "0")).png", img)
    end

end


# loading data from exp_0 dataset
function load_from_file(filename, trial)
	file = h5open(filename, "r")
	dataset = read(file, "dataset")
	data = dataset["$(trial-1)"]

	obs = data["obs"]
	avg_vel = data["avg_vel"]
	dots = data["gt_dots"]
	init_dots = data["init_dots"]
		
	inertia = data["inertia"]
	spring = data["spring"]
	sigma_w = data["sigma_w"]	
	sigma_x = data["sigma_x"]	
	sigma_v = data["sigma_v"]	

	# adding measurement noise to simulate the perception module
	stds = fill(2.0, size(obs))
	obs = broadcasted_normal(obs, stds)

    # adding z layer for optics
    new_obs = []
    new_dots = []
    for t=1:size(obs,1)
        t_obs = []
        t_dots = []
        for i=1:size(obs,2)
            push!(t_obs, [obs[t,i,:] ; 0.5])
            push!(t_dots, [dots[t,i,:] ; 0.5])
        end
        push!(new_obs, t_obs)
        push!(new_dots, t_dots)
    end
	
	return new_obs, avg_vel, new_dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v
end

function test()
    #Random.seed!(4)

    # starting the profiler
    #statprofilehtml()

    ##################################
    # getting data from previous exp_0
    T = 120
    num_particles = 10
    num_observations = 8
    num_trackers = 4
    num_distractors_rate = 4

    dataset = "datasets/exp_0.h5"
    trial = 11
    obs, avg_vel, dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v = load_from_file(dataset, trial)
    position_noise = 2.0
    depth_noise = 0.5

    rejuv_smoothness = 1.03
    max_rejuv = 0

    attended_trackers = fill([], T)

    obs = obs[1:T]
    dots = dots[1:T]
    
	params = Params(inertia, spring, sigma_w, sigma_v,
                    position_noise, depth_noise,
                    num_trackers, num_distractors_rate,
                    rejuv_smoothness,
                    max_rejuv,
                    800,
                    800,
                    attended_trackers)
                    

    ################################# 
    
    stimuli = false
    if stimuli
        overlay(dots, params.num_trackers; folder_name="stimuli", stimuli=true, highlighted=[1,2,3,4])
        return
    end
    
    
    #obs, init_dots = stress_test_0(T)
    #default_params.attended_trackers = fill([], T)
    #default_params.max_rejuv = 0
    #default_params.position_noise = 5.0
    #default_params.inertia = 0.8
    #default_params.sigma_w = 1.0

    results = run_inference(obs, init_dots, params, T, num_particles)
    extracted = extract_chain(results)

    xy = extracted["unweighted"][:tracker_positions]
    
    # extracting attended in a nicer form
    attended = zeros(T, num_trackers)
    rej_moves = zeros(Int, T)
    for t=1:T
        att = default_params.attended_trackers[t]
        for a in att
           attended[t,a] += 1 
           rej_moves[t] += 1
        end
    end
    attended /= max_rejuv*num_particles
    rej_moves /= num_particles
    
    plot_rejuvenation(rej_moves)
    plot_xy(xy)

    overlay(obs, params.num_trackers; pf_xy=xy, stimuli=false, highlighted=[1,2,3,4], attended=attended)

    return extracted
end

#revise(MOT)
Gen.load_generated_functions()
test()
