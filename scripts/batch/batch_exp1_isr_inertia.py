#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
def tavg_tasks(args):
    tasks = [(t,c) for c in range(1, args.chains + 1) 
             for t in range(1, args.scenes+1)]
    return (tasks, [], [])

import os
import argparse
from slurmpy import sbatch

script = 'bash {0!s}/run.sh julia -C "generic" -J /project/mot.so ' + \
         '/project/scripts/inference/isr_inertia/isr_inertia.jl'

def att_tasks(args):
    tasks = [(t,c,args.att_key) for c in range(1, args.chains + 1) 
             for t in range(1, args.scenes+1)]
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scenes', type = int, default = 60,
                        help = 'number of scenes')
    parser.add_argument('--chains', type = int, default = 20,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 60,
                        help = 'job duration (min)')

    subparsers = parser.add_subparsers(title='Attention models')

    parser_td = subparsers.add_parser('td', help='Using target designation')
    parser_td.set_defaults(func=att_tasks, att_key = 'T')

    parser_dc = subparsers.add_parser('dc', help='Using data correspondence')
    parser_dc.set_defaults(func=att_tasks, att_key = 'D')

    parser_ta = subparsers.add_parser('ta', help='Using trial avg')
    parser_ta.add_argument('model', type = str, help='Exp run for compute')
    parser_ta.set_defaults(func=att_tasks, att_key = 'A')

    args = parser.parse_args()

    n = args.scenes * args.chains
    tasks, kwargs, extras = args.func(args)

    interpreter = '#!/bin/bash'
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '4GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'short',
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
