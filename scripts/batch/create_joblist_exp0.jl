file = open("joblist.txt", "w")
beginning = "module load singularity; "
script = "scripts/batch/run_exp0.jl"

trials = [x for x=1:128]
num_runs = 20

attention = false

if attention
	compute_types = ["none"]
else
	compute_types = ["trial_avg", "base"]
end

for compute_type in compute_types
	for trial in trials
		for run=1:num_runs
			cmd = "./run.sh julia $script "
			cmd *= "$run "
			cmd *= "$attention "
			cmd *= "$trial "
			cmd *= "$compute_type "
			println(file, cmd)
			println(cmd)
		end
	end
end

close(file)
