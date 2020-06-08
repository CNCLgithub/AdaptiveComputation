using MOT

using HDF5
using Gadfly
using Compose
using GLM
using HypothesisTests
using Statistics
using DataFrames


# extracts localization error and distance to nearest distractor
# from loc_error dictionary
# (making sure that we only consider targets that are correctly identified)
function extract_data(loc_error, rejuvenation)
	distances = []
	errors = []
	compute = []
	assignment = []
	
	map_assignment = loc_error["map_assignment"]
	map_positions = loc_error["map_positions"]
	gt_targets = loc_error["gt_targets"]
	distances_nd = loc_error["distances_nd"]
	num_particles = loc_error["num_particles"]
	avg_rej = loc_error["avg_rej"]

	num_targets = length(map_assignment)
	targets = collect(1:num_targets)
	got_all_correctly = intersect(targets, map_assignment) == targets ? true : false
	if got_all_correctly
		for i=1:num_targets
			assigned = Int(map_assignment[i])
			if assigned in collect(1:num_targets)
				target = map_positions[i,:]
				gt_target = gt_targets[assigned,:]
				error = dist(target, gt_target)
				push!(distances, distances_nd[i])
				push!(errors, error)
				c = rejuvenation ? num_particles + num_particles*avg_rej : num_particles
				push!(compute, c)
			end
		end
	end
	
	#for i=1:num_targets
	#	assigned = Int(map_assignment[i])
	#	if assigned in collect(1:num_targets)
	#		target = map_positions[i,:]
	#		gt_target = gt_targets[assigned,:]
	#		error = dist(target, gt_target)
	#		push!(distances, distances_nd[i])
	#		push!(errors, error)
	#		push!(assignment, assigned)
	#	end
	#end

	distances = convert(Vector{Float64}, distances)
	errors = convert(Vector{Float64}, errors)
	compute = convert(Vector{Float64}, compute)
	#compute = rejuvenation ? num_particles + num_particles*avg_rej : num_particles

	return distances, errors, compute #, assignment, length(assignment)
end


function plot_localization_error(df, num_particles, num_rejuv, rejuvenation, measurement_noise; no_rejuv_type="rejuv")
	if rejuvenation
		title = "REJUVENATION, $num_particles particles, $num_rejuv rej moves,
					measurement_noise $measurement_noise"
	else
		title = "NO REJUVENATION equivalent"
	end
	

	println(df)

	# arbitrary compute restriction
	#df = df[df[:compute] .> 77.5, :]
	#df = df[df[:compute] .< 82.5, :]

	
	# making a copy of the dataframe for the cumulative plot
	df_cumu = deepcopy(df)

	# avg distance plot
	num_bins = 6


	dist_nd = collect(LinRange(minimum(df[:distances_nd]), maximum(df[:distances_nd]), num_bins+1))
	dist_nd .+= (dist_nd[2] - dist_nd[1])/2
	dist_nd = dist_nd[1:end-1]
	df.distances_nd = (x -> dist_nd[findnearest(dist_nd, x)][1]).(df.distances_nd)

	loc_error_mean = Vector{Float64}(undef, num_bins)
	loc_error_min = Vector{Float64}(undef, num_bins)
	loc_error_max = Vector{Float64}(undef, num_bins)

	for (i, distance) in enumerate(dist_nd)
		df_subset = df[df[:distances_nd] .== distance, :]
		conf_int = ci(OneSampleTTest(df_subset.localization_error))
		loc_error_mean[i] = mean(df_subset.localization_error)
		
		loc_error_min[i] = conf_int[1]
		loc_error_max[i] = conf_int[2]
	end
	

	ylabel = "localization error (MAP estimate, pixels)"

	ymin_plot = 5.0
	ymax_plot = 12.0

	avg_plot = plot(x=dist_nd,
			 y=loc_error_mean,
			 ymin=loc_error_min,
			 ymax=loc_error_max,
			 Geom.line, Geom.errorbar,
			 Scale.y_continuous(minvalue=ymin_plot, maxvalue=ymax_plot),
			 Guide.xlabel("avg D_nearest distractor (pixels)"),
			 Guide.ylabel(ylabel),
			 Guide.title(title),
			 Theme(background_color="white"))

	# plotting cumulative
	num_bins = 15
	skip_first = 1

	dist_nd = collect(LinRange(minimum(df_cumu[:distances_nd]), maximum(df_cumu[:distances_nd]), num_bins+skip_first))
	dist_nd = dist_nd[1+skip_first:end]

	loc_error_mean = Vector{Float64}(undef, num_bins)
	loc_error_min = Vector{Float64}(undef, num_bins)
	loc_error_max = Vector{Float64}(undef, num_bins)
	
	for (i, distance) in enumerate(dist_nd)
		df_subset = df_cumu[df_cumu[:distances_nd] .< distance, :]
		conf_int = ci(OneSampleTTest(df_subset.localization_error))
		loc_error_mean[i] = mean(df_subset.localization_error)

		loc_error_min[i] = conf_int[1]
		loc_error_max[i] = conf_int[2]
	end
	

	max_plot = plot(x=dist_nd,
			 y=loc_error_mean,
			 ymin=loc_error_min,
			 ymax=loc_error_max,
			 Geom.line, Geom.errorbar,
			 Scale.y_continuous(minvalue=ymin_plot, maxvalue=ymax_plot),
			 Guide.xlabel("max D_nearest distractor (pixels)"),
			 Guide.ylabel(ylabel),
			 Guide.title(title),
			 Theme(background_color="white"))
	return avg_plot, max_plot
end

function extract_scene_data(path, rejuvenation)
		hdf5s = readdir(path)
	
		distances = []
		localization_error = []
		compute = []
		assignment = []
		performance = []


		for hdf5 in hdf5s
			h = h5open("$(path)/$(hdf5)", "r") do file
				loc_error = read(file, "loc_error")
				d, loc, comp, a, perf = extract_data(loc_error, rejuvenation)
				push!(distances, d)
				push!(localization_error, loc)
				push!(compute, comp)
				push!(assignment, a)
				push!(performance, perf)
			end
		end
		
		println(path)
		println(performance)
		if length(performance) > 0 && mean(performance) > 2.7
			println(mean(performance))
			localization_error_result = zeros(3)
			distances_result = zeros(3)
			count = zeros(3)

			for (i, locs) in enumerate(localization_error)
				for (j, loc) in enumerate(locs)
					assigned = assignment[i][j]
					distance = distances[i][j]
					localization_error_result[assigned] += loc
					distances_result[assigned] += distance
					count[assigned] += 1
					println(count)
				end
			end
			for i=1:3
				localization_error_result[i] /= count[i]
				distances_result[i] /= count[i]
			end

			return distances_result, localization_error_result, fill(mean(compute), 3)
		else
			return nothing, nothing, nothing
		end
end

function _prepare_data(num_particles, num_rejuv, rejuvenation, measurement_noise; no_rejuv_type="rejuv")
	final_df = DataFrame()
	dir_path = "hdf5/old_hdf5s/measurement_noise_$(measurement_noise)/$(num_particles)p_$(num_rejuv)r/"
	dir_path *= rejuvenation ? "rejuv/" : "no_rejuv/"
	#dir_path *= !rejuvenation ? "$(no_rejuv_type)/" : ""

	trials = readdir(dir_path)
	
	trials = [""]

	for trial in trials
		path = "$(dir_path)$(trial)"
		hdf5s = readdir(path)
		for hdf5 in hdf5s
			h = h5open("$(path)/$(hdf5)", "r") do file
			loc_error = read(file, "loc_error")
			distances_nd, localization_error, compute = extract_data(loc_error, rejuvenation)
			df = DataFrame(distances_nd=distances_nd, localization_error=localization_error,
						  compute=compute)
			if size(df.distances_nd, 1) > 0 && maximum(df.distances_nd) > 0
				final_df = vcat(final_df, df)
			end
		end
		end
		#distances_nd, localization_error, compute = extract_scene_data("$(dir_path)$(trial)", rejuvenation)
		#if typeof(distances_nd) == Nothing continue end
	end
	return final_df
end


function particles_rejuv_composition(num_particles, num_rejuv, measurement_noise)
	df_rejuv = _prepare_data(num_particles, num_rejuv, true, measurement_noise)
	avg_plot_rejuv, max_plot_rejuv = plot_localization_error(df_rejuv, num_particles, num_rejuv, true, measurement_noise; no_rejuv_type="rejuv")

	df_no_rejuv = _prepare_data(num_particles, num_rejuv, false, measurement_noise)
	avg_plot_no_rejuv, max_plot_no_rejuv = plot_localization_error(df_no_rejuv, num_particles, num_rejuv, false, measurement_noise)
	
	stack = gridstack([max_plot_rejuv max_plot_no_rejuv;
					   avg_plot_rejuv avg_plot_no_rejuv])
	
	return stack
end


p = particles_rejuv_composition(50, 5, 5.0)
draw(SVG("localization_error.svg", 15Gadfly.inch, 10Gadfly.inch), p)
