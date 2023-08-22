#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch

script = 'bash {0!s}/env.d/run.sh julia ' + \
         '/project/scripts/inference/exp1_difficulty/exp1_difficulty.jl'

def att_tasks(args):
    tasks = [('--scene {0:d}'.format(t),
              '--chain {0:d}'.format(c)) for c in range(1, args.chains + 1)
             for t in range(1, args.scenes+1)]
    return (tasks, [], [])

def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1 (Effort)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scenes', type = int, default = 65,
                        help = 'number of scenes')
    parser.add_argument('--chains', type = int, default = 20,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 35,
                        help = 'job duration (min)')

    args = parser.parse_args()

    #  tasks, kwargs, extras = fig4_tasks(args)
    tasks, kwargs, extras = att_tasks(args)
    n = len(tasks)

    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '3GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'psych_scavenge',
        'requeue' : None,
        'job-name' : 'mot',
        'chdir' : os.getcwd(),
        'output' : os.path.join(os.getcwd(), 'env.d/spaths/slurm/%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=n)))
    batch.run(n = n, check_submission = False)

if __name__ == '__main__':
    main()
