using HDF5

# takes directory of a particular trial with multiple runs
# returns particle filter state, assignment, ground truth state, expected performance, compute used
function load_results(dir)
	hdf5s = readdir(dir)
	
	file = h5open("$(dir)/$(first(hdf5s))", "r")
	data = read(file, "runs")
	T = size(data["assignments"], 1)
	
	close(file)

	runs = length(hdf5s)
    println(dir)
    println(runs)

	pf = Array{Float64}(undef, runs, T, num_targets, 2)
	A = Array{Int}(undef, runs, T, num_targets)
	gt_dots = Array{Float64}(undef, T, num_observations, 2)
	performance = Vector{Float64}(undef, runs)
	compute = Vector{Float64}(undef, runs)
    

	for run=1:runs
		file = h5open("$(dir)/$(hdf5s[run])", "r")
		data = read(file, "runs")
		pf[run,:,:,:] = data["pf"]
		A[run,:,:] = data["assignments"]
		gt_dots = data["gt_dots"]
		performance[run] = data["performance"]
		compute[run] = data["compute"]
		close(file)
	end

	return pf, A, gt_dots, performance, compute
end



# like load_results(dir), but for all the trials in dir
function load_results_trials(dir)
	trials = readdir(dir)
	num_trials = length(trials)

	hdf5s = readdir("$dir/$(first(trials))")
	runs = length(hdf5s)
	file = h5open("$(dir)/$(first(trials))/$(first(hdf5s))", "r")
	data = read(file, "runs")
	num_particles = size(data["assignments"], 2)
	T = size(data["assignments"], 1)
	close(file)


	pf = Array{Float64}(undef, num_trials, runs, T, num_targets, 2)
	A = Array{Int}(undef, num_trials, runs, T, num_targets)
	gt_dots = Array{Float64}(undef, num_trials, T, num_observations, 2)
	performance = Array{Float64}(undef, num_trials, runs)
	compute = Array{Float64}(undef, num_trials, runs)
	
	for trial=1:num_trials
		pf[trial,:,:,:,:], A[trial,:,:,:], gt_dots[trial,:,:,:], performance[trial,:], compute[trial,:] = load_results("$dir/$trial")
	end

	return pf, A, gt_dots, performance, compute
end
