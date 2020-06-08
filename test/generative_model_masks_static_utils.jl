function draw_image(masks, params)
    img = BitArray{2}(undef, params.img_height, params.img_width)
    img .= false

    for mask in collect(masks)
        img = img .|= mask
    end

    return img
end

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
	#stds = fill(2.0, size(obs))
	#obs = broadcasted_normal(obs, stds)

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


function visualize(xy, full_imgs, params, T, num_particles)
    # just some non file magic kind of thing
    
    for t=1:length(full_imgs)
        img = full_imgs[t]

        for p=1:num_particles
            for i=1:size(xy,3)
                x = xy[t,p,i,1]
                y = xy[t,p,i,2]
                x, y = translate_area_to_img(x, y, params)

                draw_circle!(img, [x,y], 5.0, false)
                draw_circle!(img, [x,y], 3.0, true)
                draw_circle!(img, [x,y], 1.0, false)
                #draw_circle!(img, [x,y], 5.0, 0.0)
                #draw_circle!(img, [x,y], 3.0, 1.0)
                #draw_circle!(img, [x,y], 1.0, 0.0)
            end
        end

        save("full_imgs/$(lpad(t, 3, "0")).png", img)
    end

end

function get_full_imgs(T, choices, params)
    full_imgs = []
    for t=1:T
        masks = choices[:states => t => :masks]
        for m=1:length(masks)
            save("masks/$(lpad(t, 3, "0"))_$(lpad(m, 3,"0")).png", masks[m])
        end
        img = draw_image(masks, params)
        push!(full_imgs, img)
    end
    return full_imgs
end
