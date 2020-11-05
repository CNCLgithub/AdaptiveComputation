#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
def tavg_tasks(args):
    tasks = [(t,) for t in range(1, args.scenes+1)]
    return (tasks, [], [])


import os
import argparse
from slurmpy import sbatch

script = 'bash {0!s}/run.sh julia -J /project/mot.so ' + \
         '/project/scripts/stimuli/probes.jl'

default_probe_map = "/datasets/isr_inertia_probe_map_nd.csv"
default_dataset_path = "/datasets/exp1_isr.jld2"

def gen_tasks(args):
    tasks = [(args.probe_map, args.dataset_path, i) for i in range(1, args.scenes+1)]
    return (tasks, [], [])

def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for rendering probes',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--probe_map', type = str, default = default_probe_map,
                        help = 'Probe map for scene x tracker x epoch')
    parser.add_argument('--dataset_path', type = str, default = default_dataset_path,
                        help = 'Dataset path for scenes')
    parser.add_argument('--scenes', type = int, default = 12,
                        help = 'Number of scenes')

    args = parser.parse_args()

    n = args.scenes
    duration = 30 # in minutes
    tasks, kwargs, extras = gen_tasks(args)

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
