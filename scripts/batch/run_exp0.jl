using MOT
using Gen
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
	
	return obs, avg_vel, dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v
end

# type can be "rejuv" (based on individual rejuv trial), "avg" or "max"
function get_compute(type)
	# for testing purposes!
	println("returning 10000 particles")
	return 1000

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


function run_pf(run, rejuvenation, dataset, trial; compute_type="")

	rejuv_count = 10
	measurement_noise = 2.0

	(obs, avg_vel, gt_dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v) = load_from_file(dataset)
	
	if rejuvenation != true
		# changing for testing
		rejuv_count = 10
		global num_particles = round(Int, get_compute(compute_type))
	end
	params = Params(inertia, spring, sigma_w, sigma_x, sigma_v,
				measurement_noise, num_targets, num_observations, rejuv_count)
	println(params)
	println(rejuv_count)
	println(num_particles)
	
	pf = Array{Float64}(undef, T, num_targets, 2)
	A = Array{Int}(undef, T, num_targets)

	println(size(pf))
	println("running: $run")
	(state, log_weights, pf_xy, pf_xy_unweighted, rejuvenations, assignments, final_assignment) = particle_filter(generative_model, num_particles, obs, init_dots, params)
	for t=1:T
		p = argmax(log_weights[t,:])
		pf[t,:,:] = pf_xy[t,p,:,:]
	end

	performance = 0.0
	for p=1:num_particles
		performance += length(intersect(final_assignment[p,:], 1:num_targets))	
	end
	performance /= num_particles
	println(performance)

	return pf, assignments, gt_dots, rejuvenations, num_particles, performance
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
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
trial = args["trial"]
rejuvenation = args["rejuvenation"]
compute_type = args["compute_type"]
compute_type *= "_1000_rejuv"


dataset = "datasets/exp_0.h5"

T = 120
num_particles = 10
num_observations = 8
num_targets = 4

rejuv = rejuvenation ? "rejuv" : "no_rejuv_$(compute_type)"
folder = "exp0_results/$(rejuv)/$(trial)"
mkpath(folder)
path = "$folder/$run.h5"
if ispath(path)
	error("file exists, exiting..")
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
