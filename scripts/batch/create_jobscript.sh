#!/bin/bash

./run.sh julia scripts/batch/create_joblist_exp0.jl

module load dSQ
PARTITION=short

mkdir -p std_out
dsq --job-file joblist.txt --partition $PARTITION --requeue --cpus-per-task 1 --mem-per-cpu 2gb -t 20 -o std_out/dsq-jobfile-%A_%a-%N.out --batch-file jobscript.sh
wc -l joblist.txt
