#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch

script = 'bash {0!s}/run.sh julia -C "generic" -J /project/mot.so ' + \
         '/project/scripts/inference/ia_rf/ia_rf.jl'

def get_tasks(args):
    tasks = [(scene, chain, compute, n_targets)
            for n_targets in [3, 5]
            for compute in range(1, args.compute+1)
            for chain in range(1, args.chains+1) 
            for scene in range(1, args.scenes+1)]
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp Individual Attention (Receptive Fields)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scenes', type = int, default = 56,
                        help = 'number of scenes')
    parser.add_argument('--chains', type = int, default = 20,
                        help = 'number of chains')
    parser.add_argument('--compute', type = int, default = 30,
                        help = 'max compute value')
    parser.add_argument('--duration', type = int, default = 60,
                        help = 'job duration (min)')
    parser.set_defaults(func=get_tasks)
    args = parser.parse_args()

    n = args.scenes * args.chains * args.compute * 2
    tasks, kwargs, extras = args.func(args)


    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '1GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'scavenge',
        'requeue' : None,
        'job-name' : 'mot',
        'output' : os.path.join(os.getcwd(), 'output/slurm/%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=n)))
    batch.run(n = n, check_submission = False)

if __name__ == '__main__':
    main()
