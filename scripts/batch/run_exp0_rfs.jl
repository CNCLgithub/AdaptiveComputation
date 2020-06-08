using MOT
using Gen
using Gen_Compose
Gen.load_generated_functions()

using HDF5
using Gadfly

using ArgParse
using Base.Filesystem
using Statistics


function save_result(filename, pf, assignments, gt_dots, compute, performance, rejuvenations)
	h5open(filename, "w")  do file
		runs = g_create(file, "runs")
		runs["pf"] = pf
		runs["assignments"] = assignments
		runs["gt_dots"] = gt_dots
		runs["compute"] = compute
		runs["performance"] = performance
		runs["rejuvenations"] = rejuvenations
	end
end

function load_from_file(filename)
	file = h5open(filename, "r")
	dataset = read(file, "dataset")
	data = dataset["$(trial-1)"]

	obs = data["obs"]
	avg_vel = data["avg_vel"]
	gt_dots = data["gt_dots"]
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
    for t=1:size(obs,1)
        t_obs = []
        for i=1:size(obs,2)
            push!(t_obs, [obs[t,i,:] ; 0.5])
        end
        push!(new_obs, t_obs)
    end
	
	return new_obs, avg_vel, gt_dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v
end


# type can be "rejuv" (based on individual rejuv trial), "avg" or "max"
function get_compute(type)
	dir = "exp0_results/rejuv"
	trials = readdir(dir)
	hdf5s = readdir("$dir/$trial")
	
	num_trials = length(trials)	
	num_runs = length(hdf5s)

	compute = zeros(num_trials, num_runs)
	
	for i=1:num_trials
		for j=1:num_runs
			file = h5open("$(dir)/$(i)/$(j).h5", "r")
			data = read(file, "runs")
			compute[i,j] += data["compute"]	
		end
	end	
	
	if type=="trial"
		return mean(compute[trial,:])
	elseif type=="avg"
		return mean(compute)
	elseif type=="max"
		averages = mean(compute, dims=2)
		return maximum(averages)
	elseif type=="base"
		return num_particles
	else
		error("unknown get compute type")
	end
end

function run_inference(obs, init_dots, params, T, num_particles)
    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions,
                                :assignments => extract_assignments
                               ))

    init_obs = Gen.choicemap()
    for i=1:params.num_trackers
        addr = :init_state => :init_trackers => i => :x
        init_obs[addr] = init_dots[i,1] #choices[addr]
        addr = :init_state => :init_trackers => i => :y
        println(init_dots)
        init_obs[addr] = init_dots[i,2] #choices[addr]
    end
    
    args = [(t, params) for t in 1:T]
    observations = Vector{Gen.ChoiceMap}(undef, T)
    for t = 1:T
        cm = Gen.choicemap()
        cm[:states => t => :optics] = obs[t]
        observations[t] = cm
    end

    query = Gen_Compose.SequentialQuery(latent_map,
                                        generative_model_masks_static,
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

    results = sequential_monte_carlo(procedure, query,
                                     buffer_size = T,
                                     path = nothing)
    
    return results
end


function run_pf(run, rejuvenation, dataset, trial; compute_type="")

	max_rejuv = 10
    rejuv_smoothness = 1.03

    position_noise = 2.0
    depth_noise = 0.0
    
    # just a parameter to track which trackers are attended
    attended_trackers = fill([], T)

	(obs, avg_vel, gt_dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v) = load_from_file(dataset)
	
	if rejuvenation != true
	    max_rejuv = 0
		global num_particles = round(Int, get_compute(compute_type))
	end

	#params = Params(inertia, spring, sigma_w, sigma_x, sigma_v,
	#			measurement_noise, num_trackers, num_observations, rejuv_count)

	params = Params(inertia, spring, sigma_w, sigma_v,
                    position_noise, depth_noise,
                    num_trackers, num_distractors_rate,
                    rejuv_smoothness,
                    max_rejuv,
                    800,
                    800,
                    attended_trackers)

	println(params)
	println(max_rejuv)
	println(num_particles)
	
	pf = Array{Float64}(undef, T, num_trackers, 2)
	A = Array{Int}(undef, T, num_trackers)

	println(size(pf))
	println("running: $run")
    results = run_inference(obs, init_dots, params, T, num_particles)
	#(state, log_weights, pf_xy, pf_xy_unweighted, rejuvenations, assignments, final_assignment) = particle_filter(generative_model, num_particles, obs, init_dots, params)
    
    extracted = extract_chain(results)
    pf_xy = extracted["unweighted"][:tracker_positions]
    assignments = extracted["unweighted"][:assignments]
    final_assignment = assignments[end,:,:] # storing final assignments for all particles
    assignments = assignments[:,1,:] # picking first tracker designation

    println("assignments: $assignments")
    println("final assignment: $final_assignment")

	for t=1:T
		#p = argmax(log_weights[t,:])
        p = 1 # doesn't matter because unweighted
		pf[t,:,:] = pf_xy[t,p,:,:]
	end

    # extracting rejuvenations TODO make it nicer
    rejuvenations = zeros(T)
    for t=1:T
        att = params.attended_trackers[t]
        for a in att
           rejuvenations[t] += 1
        end
    end
    rejuvenations /= num_particles

	performance = 0.0
	for p=1:num_particles
		performance += length(intersect(final_assignment[p,:], 1:num_trackers))	
	end
	performance /= num_particles
	println(performance)

	return pf, assignments, gt_dots, rejuvenations, num_particles, performance
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "run"
            help = "run"
		arg_type = Int
            required = true
	"rejuvenation"
		arg_type = Bool
		required = true
        "trial"
		arg_type = Int
            required = true
	"compute_type"
		arg_type = String
            required = true
    end

    return parse_args(s)
end

args = parse_commandline()

run = args["run"]
#run = 9

trial = args["trial"]
#trial = 8

rejuvenation = args["rejuvenation"]
#rejuvenation = true

compute_type = args["compute_type"]
#compute_type = "rejuv"

dataset = "datasets/exp_0.h5"

T = 120
num_particles = 10
num_observations = 8
num_trackers = 4
num_distractors_rate = num_observations - num_trackers
    

rejuv = rejuvenation ? "rejuv" : "no_rejuv_$(compute_type)"
folder = "exp0_results/$(rejuv)/$(trial)"
mkpath(folder)
path = "$folder/$run.h5"
if ispath(path)
	#error("file exists, exiting..")
end

pf, A, gt_dots, rejuvenations, num_particles, performance = run_pf(run, rejuvenation, dataset, trial; compute_type=compute_type)
compute = num_particles + num_particles*mean(rejuvenations)


println("pf: $pf")
println("A: $A")
println("gt_dots: $gt_dots")
println("compute: $compute")
println("performance: $performance")
println("rejuvenations: $rejuvenations")
save_result(path, pf, A, gt_dots, compute, performance, rejuvenations)
