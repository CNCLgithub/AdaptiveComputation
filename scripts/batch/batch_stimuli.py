#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
from slurmpy import sbatch


pwd = os.environ['PWD']
func = 'cd {0!s} && ./run.sh julia scripts/render_scene.jl'.format(pwd)


def submit_batch(dataset, n, max_size = 32):

    njobs = min(n, max_size)
    duration = 120 # in minutes

    interpreter = '#!/bin/bash'
    tasks = [(dataset, t) for t in range(n)]
    kargs= []
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '2GB',
        'time' : '{0:d}'.format(duration),
        'partition' : 'scavenge',
        'requeue' : None,
        # 'output' : os.path.join('/sout',
        #                        'slurm-%A_%a.out')
    }
    os.environ['PATH'] += ':/project'
    batch = sbatch.Batch(interpreter, func, tasks, kargs, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=njobs)))
    batch.run(n = njobs, check_submission = False)

def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs to render stimuli.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset', type = str,
                        help = 'path to scene files')
    parser.add_argument('--n', type = int, default = 128,
                        help = 'number of trials to run')
    args = parser.parse_args()

    submit_batch(args.dataset, args.n)


if __name__ == '__main__':
    main()
