using MOT
using Gen
#Gen.load_generated_functions()

using HDF5
using Gadfly
using Statistics
using GLM
using DataFrames
using HypothesisTests

include("loading_utils.jl")

#### parameters ####
T = 120
first_timestep = 11

num_particles = 10
num_rejuv = 10

num_observations = 8
num_targets = 4

maximum_distance = 150

plot_dir = "plots/loc_error_plots"
results_dir = "exp0_results"
rejuvenation = false
compute_type = "avg"
####################


"""
Main function that does the plotting
"""
function plot_loc_error(df, avg_performance, compute; calculate_mean=true, only_correct=false, minvalue=0.0, maxvalue=10.0)
	
	title = rejuvenation ? "REJUV" : "NO REJUV"
	title *= calculate_mean ? " MEAN" : " VARIANCE"
	title *= only_correct ? "only correct" : "all"
	#title *= " (particles $num_particles, rejuv $num_rejuv, avg performance $avg_performance, avg compute $compute)"
	title *= " (avg performance $avg_performance, avg compute $compute)"
	
	# avg distance plot
	num_bins = 8

	dist_nd = collect(LinRange(0.0, maximum_distance, num_bins+1))
	dist_nd .+= (dist_nd[2] - dist_nd[1])/2
	dist_nd = dist_nd[1:end-1]
	df.dist_nd = (x -> dist_nd[findnearest(dist_nd, x)][1]).(df.dist_nd)

	loc_error_mean = Vector{Float64}(undef, num_bins)
	loc_error_min = Vector{Float64}(undef, num_bins)
	loc_error_max = Vector{Float64}(undef, num_bins)

	xlabel = "avg D_nearest (pixels), [bin counts "

	for (i, distance) in enumerate(dist_nd)
		df_subset = df[df[:dist_nd] .== distance, :]
		xlabel *= "$(nrow(df_subset))"
		if i != length(dist_nd) xlabel *= ", " end
		# TODO change to bootstrap
		conf_int = HypothesisTests.ci(OneSampleTTest(df_subset.loc_error))

		loc_error_mean[i] = mean(df_subset.loc_error)
		loc_error_min[i] = conf_int[1]
		loc_error_max[i] = conf_int[2]
	end

	xlabel *= "]"

	mkpath(plot_dir)
	filename = "loc_error_"
	filename *= "$(calculate_mean ? "mean" : "var")_"
	filename *= "$(only_correct ? "correct" : "all").svg"
	path = "$plot_dir/$filename"

	p = plot(x=dist_nd,
			 y=loc_error_mean,
			 ymin=loc_error_min,
			 ymax=loc_error_max,
			 Geom.line, Geom.errorbar,
			 Scale.y_continuous(minvalue=minvalue, maxvalue=maxvalue),
			 Guide.title(title),
			 Guide.xlabel(xlabel),
			 Guide.ylabel("localization error (pixels)"),
			 Theme(background_color="white"))
	Gadfly.draw(SVG(path, 7Gadfly.inch, 9Gadfly.inch), p)
end



"""
returns a matrix of localization errors and distances to nearest distractor

if calculate_mean, then returns mean, otherwise returns variance

if only_correct, then includes only correctly assigned trackers,
otherwise includes all trackers
"""
function calculate_dist_nd_loc_error(pf, A, gt_dots; calculate_mean=true, only_correct=false)

	num_dots = only_correct ? num_targets : num_observations
	
	distances_nd_mean = zeros(T, num_dots)
	localization_error_mean = zeros(T, num_dots)
	localization_error_var = zeros(T, num_dots)
	count = zeros(T, num_dots)

	runs = size(pf, 1)
	
	for t=1:T, run=1:runs
		assignment = A[run,t,:]
		for (i, assigned) in enumerate(assignment)
			if only_correct && !(assigned in 1:num_targets)
				continue
			end
			gt = gt_dots[t,assigned,:]
			pred = pf[run,t,i,:]

			complement_ix = setdiff(1:num_observations, assignment)
			distractors = gt_dots[t,complement_ix,:]

			localization_error_mean[t,assigned] += dist(gt, pred)
			distances_nd_mean[t,assigned] += find_distance_to_nd(pred, distractors)
			count[t,assigned] += 1
		end
	end
	
	for t=1:T, i=1:num_dots
		if count[t,i] == 0
			distances_nd_mean[t,i] = NaN
		else
			distances_nd_mean[t,i] /= count[t,i]
			localization_error_mean[t,i] /= count[t,i]
		end
	end
	
	if calculate_mean
		return distances_nd_mean, localization_error_mean
	end

	for t=1:T, run=1:runs
		assignment = A[run,t,:]
		for (i, assigned) in enumerate(assignment)
			if only_correct && !(assigned in 1:num_targets)
				continue
			end
			gt = gt_dots[t,assigned,:]
			pred = pf[run,t,i,:]

			complement_ix = setdiff(1:num_observations, assignment)
			distractors = gt_dots[t,complement_ix,:]

			distance = dist(gt, pred)
			loc_error_mean = localization_error_mean[t,assigned]
			squared = (distance - loc_error_mean)^2
			
			localization_error_var[t,assigned] += squared
		end
	end


	for t=1:T, i=1:num_dots
		localization_error_var[t,i] /= count[t,i]
	end
	
	return distances_nd_mean, localization_error_var
end


function calculate_dist_nd_loc_error_trials(pf, A, gt_dots; calculate_mean=true, only_correct=false)
	num_trials = size(pf, 1)
	runs = size(pf, 2)
	
	num_dots = only_correct ? num_targets : num_observations
	
	distances_nd = Array{Float64}(undef, num_trials, T, num_dots)
	localization_error = Array{Float64}(undef, num_trials, T, num_dots)

	for trial=1:num_trials
		distances_nd[trial,:,:], localization_error[trial,:,:] = calculate_dist_nd_loc_error(pf[trial,:,:,:,:], A[trial,:,:,:], gt_dots[trial,:,:,:]; calculate_mean=calculate_mean, only_correct=only_correct)
	end

	return distances_nd, localization_error
end

function _prepare_data(dist_nd, loc_error, maximum_distance)
	num_trials, timesteps, num_targets = size(dist_nd)

	distances = []
	errors = []

	for trial=1:num_trials, t=1:timesteps, i=1:num_targets
		if isnan(dist_nd[trial,t,i]) || dist_nd[trial,t,i] > maximum_distance 
			continue 
		end
		push!(distances, dist_nd[trial,t,i])
		push!(errors, loc_error[trial,t,i])
	end

	distances = convert(Vector{Float64}, distances)
	errors = convert(Vector{Float64}, errors)

	return distances, errors
end

function plot_dist_nd_loc_error_trials(dist_nd, loc_error, avg_performance, compute; calculate_mean=true, only_correct=false)

end


# helper plotting function
function _plot_loc_error(pf, A, gt_dots, performance, compute, calculate_mean, only_correct)
	dist_nd, loc_error = calculate_dist_nd_loc_error_trials(pf, A, gt_dots; calculate_mean=calculate_mean, only_correct=only_correct)

	dist_nd = dist_nd[:,first_timestep:end,:]
	loc_error = loc_error[:,first_timestep:end,:]

	dist_nd, loc_error = _prepare_data(dist_nd, loc_error, maximum_distance)
	df = DataFrame(dist_nd=dist_nd, loc_error=loc_error)
	plot_loc_error(df, mean(performance), mean(compute); calculate_mean=calculate_mean, only_correct=only_correct)
end


function loc_error()
	rejuv = rejuvenation ? "rejuv" : "no_rejuv"
	rejuv *= !rejuvenation ? "_$compute_type" : ""
	dir = "$results_dir/$rejuv"

	println("loading $dir")
	pf, A, gt_dots, performance, compute = load_results_trials(dir)

	# mean of localization error
	_plot_loc_error(pf, A, gt_dots, performance, compute, true, true)
	_plot_loc_error(pf, A, gt_dots, performance, compute, true, false)

	# variance of localization error
	_plot_loc_error(pf, A, gt_dots, performance, compute, false, true)
	_plot_loc_error(pf, A, gt_dots, performance, compute, false, false)
end

loc_error()
