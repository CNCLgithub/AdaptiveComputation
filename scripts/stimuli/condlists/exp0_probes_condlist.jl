#!/usr/bin/env python

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs to render stimuli.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--conds', type = int, default = 2,
                        help = 'Number of conditions')
    parser.add_argument('--scenes', type = int, default = 40,
                        help = 'Number of scenes')
    args = parser.parse_args()

    # create
    condlist = []
    for c in range(1, args.conds+1):
        cond_trials = []

        for s in range(1, args.scenes+1):
            idx = ((s + c) % args.conds) + 1
            trial = '{0:d}_{1:d}.mp4'.format(s, idx)
            cond_trials.append(trial)

        condlist.append(cond_trials)


    outpath = os.path.join('/renders', 'condlist.json')
    with open(outpath, 'w') as f:
        json.dump(condlist, f, indent = 4)


if __name__ == '__main__':
    main()
