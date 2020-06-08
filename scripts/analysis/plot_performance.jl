using MOT
using Gen
#Gen.load_generated_functions()

using HDF5
using Gadfly
using Statistics
using GLM
using DataFrames

using Bootstrap
using CSV


include("loading_utils.jl")

#### parameters ####
T = 120
num_trials = 128
first_timestep = 11

num_particles = 30
num_rejuv = 10

num_observations = 8
num_targets = 4

maximum_distance = 150

plot_dir = "loc_error_plots"
results_dir = "exp0_results"
models = ["rejuv", "no_rejuv_trial", "no_rejuv_avg", "no_rejuv_base"]
#models = ["rejuv"]

rejuvenation = true
compute_type = "avg"
####################



function confidence_intervals(data; samples=false)
	n_boot = 100000
	cil = 0.95
	bs = bootstrap(mean, data, BasicSampling(n_boot))
	if samples
		return bs.t1[1]
	end
	bci = confint(bs, PercentileConfInt(cil))[1]
	return bci
end


function plot_performance()
	num_targets = 4
	num_observations = 8
	num_trials = 128
	model_names = deepcopy(models)

	perf_plot = []
	perf_plot_min = []
	perf_plot_max = []

	for i=1:length(models)
		model = models[i]
		println(model)
		_, A, _, perf, comp = load_results_trials("$results_dir/$model")
		
		models[i] *= "\n compute $(mean(comp))\n perf $(mean(perf)/4)"
		compute = mean(comp; dims=2)
		println("maximum compute $(maximum(compute))")
		perf /= num_targets
		
		perf = mean(perf, dims=2)
		perf_mean, perf_min, perf_max = confidence_intervals(perf)
		push!(perf_plot_min, perf_min)
		push!(perf_plot_max, perf_max)
		push!(perf_plot, perf_mean)
	end

	bars = Gadfly.layer(x=models, y=perf_plot,
						color=model_names,
						Geom.bar)
	errorbars = Gadfly.layer(x=models,
							 y=perf_plot,
							ymin=perf_plot_min,
							ymax=perf_plot_max,
							Geom.errorbar,
							Theme(default_color="black"))

	df = DataFrame(models=model_names, y=perf_plot, ymin=perf_plot_min, ymax=perf_plot_max)
	CSV.write("models_performance.csv", df)

	performance_plots = "plots/performance_plots"
	mkpath(performance_plots)
	path = "$performance_plots/performance.svg"

	p = Gadfly.plot(errorbars, bars,
					Scale.y_continuous(minvalue=0.75, maxvalue=0.9),
					Guide.xlabel("models"),
					Guide.ylabel("performance"),
					Theme(background_color="white"))

	Gadfly.draw(SVG(path, 10Gadfly.inch, 6Gadfly.inch), p)
end

function significance_analysis(first, second)
	_,A,_,perf_1,comp = load_results_trials("$results_dir/$first")
	_,A,_,perf_2,comp = load_results_trials("$results_dir/$second")
	perf_1 = mean(perf_1, dims=2)
	perf_2 = mean(perf_2, dims=2)

	first_samples = confidence_intervals(perf_1, samples=true)
	second_samples = confidence_intervals(perf_2, samples=true)
	println(1 - count(first_samples .- second_samples .> 0)/100000)
end

function accuracy_compute()
    for model in models
        _,A,_,perf,comp = load_results_trials("$results_dir/$model")
        performance = mean(perf, dims=2)[:,1]/4
        compute = mean(comp, dims=2)[:,1]

        println(compute[99,:])
        println(mean(perf)/4)

        trials = collect(1:num_trials)

        df = DataFrame(accuracy=performance, compute=compute, trial=trials)

        mkpath("results")
        CSV.write("results/accuracy_compute_$model.csv", df)
        println(df)
    end
end

function plot_performance_between_models(model_1, model_2)
    _,_,_,perf_1,_ = load_results_trials("$results_dir/$model_1")
    _,_,_,perf_2,_ = load_results_trials("$results_dir/$model_2")

    performance_1 = mean(perf_1, dims=2)[:,1]/4
    performance_2 = mean(perf_2, dims=2)[:,1]/4

    trials = ["$i" for i=1:num_trials]
    
    p = plot(x=performance_1, y=performance_2,
             label=trials,
             Guide.xlabel(model_1),
             Guide.ylabel(model_2),
             Theme(background_color="white"),
             Geom.point,
             Geom.label)
    Gadfly.draw(PNG("performance_between_models.png", 16Gadfly.inch, 16Gadfly.inch), p)
end

#significance_analysis("rejuv", "no_rejuv_trial")
#plot_performance()
#accuracy_compute()
plot_performance_between_models("rejuv", "no_rejuv_trial")
