using MOT

include("../scripts/stimuli/difficulty.jl")

k = 8
bin = 4
n_quantiles = 3

q = Exp0(trial=124,
         k=k)
attention = MapSensitivity(samples=5,
                           sweeps=15,
                           objective=MOT.target_designation)
path = "/experiments/test"
mkpath(path)
results_path = joinpath(path, "results")
mkpath(results_path)

#results = run_inference(q, attention, path, viz=true)
diff = get_difficulty(extract_chain(results), bin)
println(diff)

function findquant(x, quantiles)
    for i=1:length(quantiles)-1
        if is_between(x, quantiles[i], quantiles[i+1])
            return i
        end
    end
end

quantiles = nquantile(collect(Iterators.flatten(diff)), n_quantiles)
difficulty = Vector{Float64}(undef, k)
for i=1:div(k, bin)
    for j=1:bin
        quant = findquant(diff[i], quantiles)
        difficulty[(i-1)*bin+j] = quant/n_quantiles
    end
end

println(difficulty)
gm = GMMaskParams(exp0=true)
positions = last(load_trial(q.trial, q.dataset_path, gm))

MOT.render(gm, dot_positions=positions[1:k],
           difficulty=difficulty)


# render_difficulty(q,
                  # results_path,
                  # joinpath(path, "difficulty"),
                  # 3,
                  # bin=4)

# compile_videos(
