using MOT: merge_experiment

# path = "/spaths/experiments/exp1_difficulty_target_designation"
# path = "/spaths/experiments/exp2_probes_adaptive_computation_td"
# path = "/spaths/experiments/exp3_staircase_adaptive_computation_td"
path = "/spaths/experiments/exp3_localization_error_adaptive_computation_td"
merge_experiment(path; report = "perf");
merge_experiment(path; report = "att");
