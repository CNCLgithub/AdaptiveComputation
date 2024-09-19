#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch

script = 'bash {0!s}/env.d/run.sh julia ' + \
         '/project/scripts/experiments/probes/exp_probes.jl'

def att_tasks(args):
    tasks = [(t,c, args.plan) for c in range(1, args.chains + 1)
             for t in range(1, args.scenes+1)]
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for probes',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--plan', type = str, default = 'ac',
                        options = ['ac', 'na'],
                        help = 'Plan objective to use')
    parser.add_argument('--scenes', type = int, default = 40,
                        help = 'number of scenes')
    parser.add_argument('--chains', type = int, default = 20,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 20,
                        help = 'job duration (min)')

    args = parser.parse_args()

    n = args.scenes * args.chains
    tasks, kwargs, extras = att_tasks(args)

    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '3GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'psych_scavenge', # Cluster specific
        'requeue' : None,
        'job-name' : 'mot-probes',
        'chdir' : os.getcwd(),
        'output' : os.path.join(os.getcwd(), 'env.d/spaths/slurm/%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    bscript = '\n'.join(batch.job_file(chunk=n, tmp_dir = 'env.d/spaths/slurm'))
    print("Template Job:")
    print(bscript)
    batch.run(n = n, check_submission = False, script = bscript)

if __name__ == '__main__':
    main()
