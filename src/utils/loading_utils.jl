export load_from_file,
        read_json

using JSON

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


"""
    read_json(path)

    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    open(path, "r") do f
        global data
        data = JSON.parse(f)
    end
    
    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end
