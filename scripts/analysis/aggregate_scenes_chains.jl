using MOT: merge_experiment

path = "/spaths/experiments/exp1_difficulty_target_designation"
# path = "/spaths/experiments/exp2_probes_target_designation"
merge_experiment(path; report = "perf");
merge_experiment(path; report = "att");
