file = open("joblist.txt", "w")
beginning = "module load singularity; "
script = "scripts/milgram/run_exp0.jl"

num_trials = 128
num_runs = 20

rejuvenation = false
if rejuvenation
	compute_types = ["none"]
else
	compute_types = ["test"]
	#compute_types = ["trial", "avg", "base"]
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
