file = open("joblist.txt", "w")
beginning = "module load singularity; "
script = "scripts/batch/run_exp0_rfs_masks.jl"

num_trials = 128
num_runs = 20

rejuvenation = true

if rejuvenation
	compute_types = ["none"]
else
	compute_types = ["trial", "avg", "base"]
end

for compute_type in compute_types
	for trial=1:num_trials
		for run=1:num_runs
			cmd = "singularity run mot.sif julia $script "
			cmd *= "$run "
			cmd *= "$rejuvenation "
			cmd *= "$trial "
			cmd *= "$compute_type "
			println(file, cmd)
			println(cmd)
		end
	end
end

close(file)
