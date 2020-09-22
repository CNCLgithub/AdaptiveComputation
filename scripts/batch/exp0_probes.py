#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
def tavg_tasks(args):
    tasks = [(t,) for t in range(1, args.scenes+1)]
    return (tasks, [], [])


import os
import argparse
from slurmpy import sbatch

script = 'bash {0!s}/run.sh julia -J /project/mot.so ' + \
         '/project/scripts/stimuli/exp0_probes.jl'

default_probe_map = "/datasets/exp0_probe_map.csv"

def att_tasks(args):
    tasks = [(t,c,args.att_key) for c in range(1, args.chains + 1)
             for t in range(1, args.trials+1)]
    return (tasks, [], [])

def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for rendering probes',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--probe_map', type = str, default = default_probe_map,
                        help = 'Probe map for scene x tracker x epoch')
    parser.add_argument('--scenes', type = int, default = 24,
                        help = 'Number of scenes')

    args = parser.parse_args()

    duration = 30 # in minutes
    tasks, kwargs, extras = args.func(args)

    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '2GB',
        'time' : '{0:d}'.format(duration),
        'partition' : 'short',
        'requeue' : None,
        'output' : os.path.join(os.getcwd(), 'slurm/%A_%a.out')
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=n)))
    batch.run(n = n, check_submission = False)

if __name__ == '__main__':
    main()
