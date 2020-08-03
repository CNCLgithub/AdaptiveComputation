#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch


base_func = 'bash {0!s}/run.sh julia -J /project/mot.so --compiled-modules=no {1!s}'

script = 'scripts/inference/exp0_trial_avg.jl'

def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for in-silico experiment.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('attention_path', type = str,
                        help = 'Experiment key')
    parser.add_argument('--trials', type = int, default = 128,
                        help = 'number of trials')
    parser.add_argument('--chains', type = int, default = 20,
                        help = 'number of chains')
    args = parser.parse_args()

    n = args.trials * args.chains
    duration = 30 # in minutes

    interpreter = '#!/bin/bash'
    tasks = [(t,c,args.attention_path) for c in range(1, args.chains + 1) 
             for t in range(1, args.trials+1)]
    kargs= []
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '2GB',
        'time' : '{0:d}'.format(duration),
        'partition' : 'short',
        'requeue' : None,
        # 'output' : os.path.join('/sout',
        #                        'slurm-%A_%a.out')
    }
    func = base_func.format(os.getcwd(), script)
    batch = sbatch.Batch(interpreter, func, tasks,
                         kargs, extras, resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=n)))
    batch.run(n = n, check_submission = False)

if __name__ == '__main__':
    main()
