using MOT

using HDF5
using Gadfly
using Statistics
#using GLM
using DataFrames
using CSV

using Bootstrap

num_trials = 128
results_dir = "exp0_results"


function performance_compute(experiments)
    for experiment in experiments
        results = load_results(joinpath(results_dir, experiment))

        perf_trial = mean(results["performance"], dims=2)[:,1]
        comp_trial = mean(results["compute"], dims=2)[:,1]
        pred_target_trial = mean(results["pred_target"], dims=2)[:,1,:]

        pred_target_packaged = []
        for i=1:size(pred_target_trial, 1)
            push!(pred_target_packaged, pred_target_trial[i,:])
        end
        
        trials = collect(1:length(perf_trial))

        df = DataFrame(performance=perf_trial, compute=comp_trial,
                       trial=trials;
                       [Symbol("dot_$i")=>pred_target_trial[:,i] for i=1:8]...
                      )

        mkpath("results")
        CSV.write("results/performance_compute_$experiment.csv", df)
        #println(df)
        println("$experiment performance: $(mean(perf_trial))")
    end
end



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


function plot_performance(models, path="plots")
	num_targets = 4
	num_observations = 8
	num_trials = 128

	perf_plot = []
	perf_plot_min = []
	perf_plot_max = []
    captions = []

	for model in models
		println(model)
        results = load_results(joinpath(results_dir, model))
		
        compute = mean(results["compute"]; dims=2)[:,1]
        perf = mean(results["performance"], dims=2)[:,1]

		perf_mean, perf_min, perf_max = confidence_intervals(perf)
		push!(perf_plot_min, perf_min)
		push!(perf_plot_max, perf_max)
		push!(perf_plot, perf_mean)
        push!(captions, model * "\n compute $(mean(compute))\n perf $(mean(perf))")
	end

	bars = Gadfly.layer(x=captions, y=perf_plot,
						color=models,
						Geom.bar)
	errorbars = Gadfly.layer(x=captions,
							 y=perf_plot,
							ymin=perf_plot_min,
							ymax=perf_plot_max,
							Geom.errorbar,
							Theme(default_color="black"))

	df = DataFrame(models=models, y=perf_plot, ymin=perf_plot_min, ymax=perf_plot_max)
	CSV.write("results/models_performance.csv", df)

	mkpath(path)

	p = Gadfly.plot(errorbars, bars,
					Scale.y_continuous(minvalue=0.75, maxvalue=0.9),
					Guide.xlabel("models"),
					Guide.ylabel("performance"),
					Theme(background_color="white"))

    Gadfly.draw(PNG(joinpath(path, "performance.png"), 10Gadfly.inch, 6Gadfly.inch), p)
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
#plot_performance_between_models("rejuv", "no_rejuv_trial")
    
models = ["attention", "trial_avg", "base"]
plot_performance(models)
#performance_compute(models)
