"""
Aggregates model chains into csv files for analysis
"""

using MOT: merge_experiment

# model = "td" # adaptive computation with target designation goal
# model = "na_perf" # no attention - matched for performance
# model = "na_load" # no attention - matched for load
model = "id" # identity tracking, for qualitative comparison

experiments = [
    # "exp_effort",
    "exp_probes",
    # "exp_localization_error",
    # "exp_staircase",
]

for exp in experiments
    path = "/spaths/experiments/$(exp)_$(model)"
    merge_experiment(path; report = "perf");
    merge_experiment(path; report = "att");
end
