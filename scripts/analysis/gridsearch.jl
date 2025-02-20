using MOT: merge_experiment

model = "ac_td" # adaptive computation with target designation goal

exp = "probes"
steps = 16

for step = 1:steps
    println("Step $(step)")
    # path = "/spaths/experiments/gridsearch_exp_$(exp)_$(model)/grid_search-$(step)"
    path = "/spaths/experiments/exp_$(exp)-gridsearch-tr/$(step)"
    merge_experiment(path; report = "perf");
    merge_experiment(path; report = "att");
end
