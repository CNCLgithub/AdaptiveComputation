#!/usr/bin/env python

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs to render stimuli.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--n_conds', type = int, default = 8,
                        help = 'Number of conditions')
    parser.add_argument('--n_scenes', type = int, default = 128,
                        help = 'Number of scenes')
    args = parser.parse_args()

    # create
    condlist = []
    for c in range(1, args.n_conds+1):
        cond_trials = []

        scene_probes = []

        for scene in range(1, args.n_scenes+1):
            cond_trials.append([scene, scene_probes])

        condlist.append(cond_trials)

    outpath = os.path.join('/renders', 'exp1_condlist.json')
    with open(outpath, 'w') as f:
        json.dump(condlist, f, indent = 4)


if __name__ == '__main__':
    main()
