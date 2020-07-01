#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch


base_func = 'bash {0!s}/run.sh julia {1!s}'

experiments = {
    'exp0_sens_td': '/project/scripts/experiments/exp0_sens_td.jl',
    # 'exp0_sens_dc': '',
    # 'exp0_entropy_td': ''
}

default_keys = list(experiments.keys())
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for in-silico experiment.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('exp_key', type = str, choices = default_keys,
                        help = 'Experiment key')
    args = parser.parse_args()

    script = experiments[args.exp_key]

    n = 128
    duration = 120 # in minutes

    interpreter = '#!/bin/bash'
    tasks = [(t,) for t in range(n)]
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
